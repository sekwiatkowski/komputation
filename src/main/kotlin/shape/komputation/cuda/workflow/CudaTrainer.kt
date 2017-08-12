package shape.komputation.cuda.workflow

import jcuda.Pointer
import shape.komputation.cuda.CudaNetwork
import shape.komputation.cuda.InputMemory
import shape.komputation.cuda.TargetMemory
import shape.komputation.cuda.loss.CudaLossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices

class CudaTrainer(
    private val network : CudaNetwork,
    private val inputs : Array<Matrix>,
    private val targets: Array<FloatArray>,
    private val numberIterations : Int,
    private val maximumBatchSize : Int,
    private val lossFunction : CudaLossFunction,
    private val afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) {

    private val numberExamples = this.inputs.size

    private val batches = partitionIndices(this.numberExamples, this.maximumBatchSize)
    private val inputMemory = InputMemory()
    private val targetMemory = TargetMemory(this.targets.first().size)

    init {

        this.lossFunction.acquire(this.maximumBatchSize)

    }

    fun free() {

        this.lossFunction.release()
        this.inputMemory.release()
        this.targetMemory.release()

    }

    fun run(): Long {

        val trackLoss = this.afterEachIteration != null

        val start = System.currentTimeMillis()

        repeat(this.numberIterations) { indexIteration ->

            var iterationLoss = if(trackLoss) 0.0f else Float.NaN

            for ((batchId, batch) in this.batches.withIndex()) {

                val currentBatchSize = batch.size

                val devicePredictions = this.network.forward(batchId, currentBatchSize, batch, this.inputs, this.inputMemory,true)
                val pointerToDevicePredictions = Pointer.to(devicePredictions)

                val pointerToTargets = this.targetMemory.get(batchId, currentBatchSize, batch, this.targets)

                if (trackLoss) {

                    this.lossFunction.accumulate(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                }

                val backwardLoss = this.lossFunction.backward(pointerToDevicePredictions, pointerToTargets, currentBatchSize)

                this.network.backward(backwardLoss, currentBatchSize)

                val scalingFactor = 1.0f.div(batch.size.toFloat())
                this.network.optimize(scalingFactor)

                if (trackLoss) {

                    val batchLoss = this.lossFunction.accessAccumulation()

                    iterationLoss += batchLoss

                }

            }

            this.afterEachIteration?.invoke(indexIteration, iterationLoss)

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        this.lossFunction.release()

        return time


    }

}