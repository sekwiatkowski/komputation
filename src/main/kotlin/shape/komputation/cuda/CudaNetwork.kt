package shape.komputation.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CudaLossFunctionInstruction
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

class CudaNetwork(entryPointInstruction: CudaEntryPointInstruction, vararg forwardLayerInstructions: CudaForwardLayerInstruction) {

    private val cudaContext = setUpCudaContext()
    private val cublasHandle = cublasHandle()

    private val entryPoint = entryPointInstruction.buildForCuda()

    private val layers = forwardLayerInstructions.map { it.buildForCuda(this.cudaContext, this.cublasHandle) }
    private val numberLayers = this.layers.size
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(batchId: Int, indices: IntArray, batchSize: Int, inputs: Array<Matrix>, isTraining: Boolean) : Pointer {

        var output = this.entryPoint.forward(batchId, indices, batchSize, inputs)

        for (layer in this.layers) {

            output = layer.forward(output, batchSize, isTraining)

        }

        return output

    }

    fun backward(lossGradient : Pointer, batchSize: Int): Pointer {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain, batchSize)

        }

        return this.entryPoint.backward(chain)

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<FloatArray>,
        loss: CudaLossFunctionInstruction,
        numberIterations : Int,
        maximumBatchSize: Int,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null): Long {

        val lossFunction = loss.buildForCuda(this.cudaContext)

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, maximumBatchSize)

        this.acquireLayerResources(maximumBatchSize)

        if (lossFunction is Resourceful) {

            lossFunction.acquire(maximumBatchSize)

        }

        val firstTarget = targets.first()
        val targetSize = firstTarget.size

        val targetMemory = hashMapOf<Int, Pointer>()

        val trackLoss = afterEachIteration != null

        val start = System.currentTimeMillis()

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = if(trackLoss) 0.0f else Float.NaN

            for ((batchId, batch) in batches.withIndex()) {

                val currentBatchSize = batch.size

                val pointerToTargets =

                    if (targetMemory.containsKey(batchId)) {

                        targetMemory[batchId]!!

                    }
                    else {

                        val batchTargetSize = maximumBatchSize * targetSize
                        val batchTargets = FloatArray(batchTargetSize)

                        for ((batchIndex, globalIndex) in batch.withIndex()) {

                            val target = targets[globalIndex]

                            System.arraycopy(target, 0, batchTargets, batchIndex * targetSize, targetSize)

                        }

                        val deviceTargets = Pointer()
                        setFloatArray(batchTargets, batchTargetSize, deviceTargets)

                        val pointerToDeviceTargets = Pointer.to(deviceTargets)

                        targetMemory[batchId] = pointerToDeviceTargets

                        pointerToDeviceTargets

                    }

                val devicePredictions = this.forward(batchId, batch, currentBatchSize, inputs, true)
                val pointerToDevicePredictions = Pointer.to(devicePredictions)

                if (trackLoss) {

                    lossFunction.accumulate(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                }

                val lossGradient = lossFunction.backward(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                this.backward(lossGradient, currentBatchSize)

                val scalingFactor = 1.0f.div(batch.size.toFloat())
                this.optimize(scalingFactor)

                if (trackLoss) {

                    val batchLoss = lossFunction.accessAccumulation()

                    iterationLoss += batchLoss

                }

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        targetMemory.values.forEach { pointer -> cudaFree(pointer) }

        if (lossFunction is Resourceful) {

            lossFunction.release()

        }

        this.releaseLayerResources()

        return time


    }

    fun optimize(scalingFactor : Float) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

        }

    }

    fun test(
        inputs: Array<Matrix>,
        targets: Array<FloatMatrix>,
        batchSize: Int,
        isCorrect: (FloatMatrix, FloatMatrix) -> Boolean) {

        val size = inputs.size

        val batches = partitionIndices(size, batchSize)

        val deviceCorrect = Pointer()
        allocateDeviceIntMemory(deviceCorrect, size)

        for ((indexBatch, batch) in batches.withIndex()) {

            val result = this.forward(indexBatch, batch, batch.size, inputs, false)

        }

    }

    fun acquireLayerResources(maximumBatchSize: Int) {

        cublasCreate(this.cublasHandle)

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire(maximumBatchSize)

        }

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire(maximumBatchSize)

            }

        }

    }

    fun releaseLayerResources() {

        cublasDestroy(this.cublasHandle)

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

        if (this.entryPoint is Resourceful) {

            this.entryPoint.release()

        }

    }

}