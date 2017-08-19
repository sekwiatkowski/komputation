package shape.komputation.cpu.workflow

import shape.komputation.cpu.CpuBackwardPropagator
import shape.komputation.cpu.CpuForwardPropagator
import shape.komputation.cpu.loss.CpuLossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

class CpuTrainer(
    private val forwardPropagator : CpuForwardPropagator,
    private val backwardPropagator : CpuBackwardPropagator,
    private val optimizables : Array<Optimizable>,
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

                    val prediction = this.forwardPropagator.forward(withinBatch, input, true)

                    val instanceLoss = this.lossFunction.forward(prediction, target)

                    val backwardInstanceLoss = this.lossFunction.backward(prediction, target)

                    this.backwardPropagator.backward(withinBatch, backwardInstanceLoss)

                    batchLoss += instanceLoss

                }

                iterationLoss += batchLoss

                val scalingFactor = 1.0f.div(batch.size.toFloat())

                for (optimizable in this.optimizables) {

                    optimizable.optimize(scalingFactor)

                }

            }

            this.afterEachIteration?.invoke(indexIteration, iterationLoss)

        }

        val stop = System.currentTimeMillis()

        val time = stop - start

        return time

    }

}