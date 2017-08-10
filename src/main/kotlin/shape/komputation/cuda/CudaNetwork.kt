package shape.komputation.cuda

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.kernels.EvaluationKernels
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.Resourceful
import shape.komputation.loss.CudaLossFunctionInstruction
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

    fun forward(batchId: Int, batchSize: Int, indices: IntArray, inputs: Array<Matrix>, inputMemory : HashMap<Int, Pointer>, isTraining: Boolean) : Pointer {

        var output = this.entryPoint.forward(batchId, batchSize, indices, inputs, inputMemory)

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
        afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null,
        memory: HashMap<Int, Pointer>? = null): Long {

        val lossFunction = loss.buildForCuda(this.cudaContext)

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, maximumBatchSize)

        val firstTarget = targets.first()
        val targetSize = firstTarget.size

        val hasSpecifiedInputMemory = memory != null

        val inputMemory = if (hasSpecifiedInputMemory) {

            memory!!

        }
        else {

            hashMapOf<Int, Pointer>()

        }

        val targetMemory = TargetMemory(targetSize, targets, maximumBatchSize)

        this.acquireLayerResources(maximumBatchSize)

        if (lossFunction is Resourceful) {

            lossFunction.acquire(maximumBatchSize)

        }

        val trackLoss = afterEachIteration != null

        val start = System.currentTimeMillis()

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = if(trackLoss) 0.0f else Float.NaN

            for ((batchId, batch) in batches.withIndex()) {

                val currentBatchSize = batch.size

                val devicePredictions = this.forward(batchId, currentBatchSize, batch, inputs, inputMemory,true)
                val pointerToDevicePredictions = Pointer.to(devicePredictions)

                val pointerToTargets = targetMemory.get(batchId, batch)

                if (trackLoss) {

                    lossFunction.accumulate(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                }

                val backwardLoss = lossFunction.backward(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                this.backward(backwardLoss, currentBatchSize)

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

        if (!hasSpecifiedInputMemory) {

            inputMemory.values.forEach { pointer ->

                cudaFree(pointer)

            }

        }

        targetMemory.release()

        val stop = System.currentTimeMillis()

        val time = stop - start

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
        targets: Array<FloatArray>,
        batchSize: Int,
        numberCategories : Int,
        length : Int = 1,
        memory : HashMap<Int, Pointer>? = null): Float {

        val numberInstances = inputs.size

        val batches = partitionIndices(numberInstances, batchSize)

        val firstTarget = targets.first()
        val targetSize = firstTarget.size

        val hasSpecifiedInputMemory = memory != null

        val inputMemory =

            if (hasSpecifiedInputMemory) {

                memory!!

            }
            else {

                hashMapOf<Int, Pointer>()

            }

        val targetMemory = TargetMemory(targetSize, targets, batchSize)

        val cudaEvaluation = CudaEvaluation(numberInstances, numberCategories, length, { this.cudaContext.createKernel(EvaluationKernels.evaluation()) })
        cudaEvaluation.acquire(batchSize)

        for ((batchId, batch) in batches.withIndex()) {

            val predictions = this.forward(batchId, batch.size, batch, inputs, inputMemory,false)

            val pointerToPredictions = Pointer.to(predictions)

            val pointerToTargets = targetMemory.get(batchId, batch)

            cudaEvaluation.evaluateBatch(batch.size, pointerToPredictions, pointerToTargets)

        }

        val accuracy = cudaEvaluation.computeAccuracy()

        cudaEvaluation.release()

        if (!hasSpecifiedInputMemory) {

            inputMemory.values.forEach { pointer ->

                cudaFree(pointer)

            }

        }

        targetMemory.release()

        return accuracy

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