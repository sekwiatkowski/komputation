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

    fun forward(id : Int, input : Matrix) : Pointer {

        var output = this.entryPoint.forward(id, input)

        for (layer in this.layers) {

            output = layer.forward(output)

        }

        return output

    }

    fun backward(lossGradient : Pointer): Pointer {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain)

        }

        return this.entryPoint.backward(chain)

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<FloatMatrix>,
        loss: CudaLossFunctionInstruction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null): Long {

        val lossFunction = loss.buildForCuda(this.cudaContext)

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, batchSize)

        this.acquireLayerResources()

        if (lossFunction is Resourceful) {

            lossFunction.acquire()

        }

        val firstTarget = targets.first()
        val targetSize = firstTarget.numberRows * firstTarget.numberColumns

        val targetMemory = hashMapOf<Int, Pointer>()

        val trackLoss = afterEachIteration != null

        val start = System.currentTimeMillis()

        repeat(numberIterations) { indexIteration ->

            var id = 0

            var iterationLoss = if(trackLoss) 0.0f else Float.NaN

            for (batch in batches) {

                for (indexExample in batch) {

                    val input = inputs[indexExample]

                    val pointerToTargets = if (targetMemory.containsKey(id)) {

                        targetMemory[id]!!

                    }
                    else {

                        val target = targets[indexExample]

                        val deviceTargets = Pointer()
                        setVector(target.entries, targetSize, deviceTargets)

                        val pointerToDeviceTargets = Pointer.to(deviceTargets)

                        targetMemory[id] = pointerToDeviceTargets

                        pointerToDeviceTargets

                    }

                    val devicePredictions = this.forward(id++, input)

                    val pointerToDevicePredictions = Pointer.to(devicePredictions)

                    if (trackLoss) {

                        lossFunction.accumulate(pointerToDevicePredictions, pointerToTargets)

                    }

                    val lossGradient = lossFunction.backward(pointerToDevicePredictions, pointerToTargets)

                    this.backward(lossGradient)

                }

                val scalingFactor = 1.0f.div(batch.size.toFloat())

                this.optimize(scalingFactor)

                if (trackLoss) {

                    iterationLoss += lossFunction.accessAccumulation()

                    lossFunction.reset()

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

    fun acquireLayerResources() {

        cublasCreate(this.cublasHandle)

        if (this.entryPoint is Resourceful) {

            this.entryPoint.acquire()

        }

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire()

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