package shape.komputation.cpu.workflow

import shape.komputation.cpu.Network
import shape.komputation.cpu.loss.CpuLossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices

class CpuTrainer(
    private val network : Network,
    private val inputs : Array<Matrix>,
    private val targets: Array<FloatArray>,
    private val numberIterations: Int,
    private val maximumBatchSize : Int,
    private val lossFunction: CpuLossFunction,
    private val afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) {

    private val batches = partitionIndices(this.inputs.size, this.maximumBatchSize)

    fun run(): Long {

        val start = System.currentTimeMillis()

        repeat(this.numberIterations) { indexIteration ->

            var iterationLoss = 0.0f

            for (batch in this.batches) {

                var batchLoss = 0.0f

                for ((withinBatch, indexExample) in batch.withIndex()) {

                    val input = this.inputs[indexExample]
                    val target = this.targets[indexExample]

                    val prediction = this.network.forward(withinBatch, input, true)

                    val instanceLoss = this.lossFunction.forward(prediction, target)

                    val backwardInstanceLoss = this.lossFunction.backward(prediction, target)

                    this.network.backward(withinBatch, backwardInstanceLoss)

                    batchLoss += instanceLoss

                }

                iterationLoss += batchLoss

                val scalingFactor = 1.0f.div(batch.size.toFloat())

                this.network.optimize(scalingFactor)

            }

            this.afterEachIteration?.invoke(indexIteration, iterationLoss)

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        return time

    }

}