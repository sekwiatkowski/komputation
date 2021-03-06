package com.komputation.cpu.workflow

import com.komputation.cpu.loss.CpuLossFunction
import com.komputation.cpu.network.CpuBackwardPropagator
import com.komputation.cpu.network.CpuForwardPropagator
import com.komputation.matrix.Matrix
import com.komputation.matrix.partitionIndices
import com.komputation.optimization.Optimizable

class CpuTrainer(
    private val forwardPropagator : CpuForwardPropagator,
    private val backwardPropagator : CpuBackwardPropagator,
    private val optimizables : Array<Optimizable>,
    private val inputs : Array<out Matrix>,
    private val targets: Array<FloatArray>,
    private val numberIterations: Int,
    private val maximumBatchSize : Int,
    private val lossFunction: CpuLossFunction,
    private val afterEachIteration : ((index : Int, loss : Float) -> Unit)? = null) {

    private val batches = partitionIndices(this.inputs.size, this.maximumBatchSize)

    fun run(): Pair<Long, Pair<List<Pair<String?, Long>>, List<Pair<String?, Long>>>> {
        val start = System.currentTimeMillis()

        repeat(this.numberIterations) { indexIteration ->

            var iterationLoss = 0.0f

            for (batch in this.batches) {

                var batchLoss = 0.0f

                for ((withinBatch, indexExample) in batch.withIndex()) {

                    val input = this.inputs[indexExample]
                    val target = this.targets[indexExample]

                    val forwardPropagation = this.forwardPropagator.forward(withinBatch, input, true)

                    val length = forwardPropagation.numberOutputColumns
                    val prediction = forwardPropagation.forwardResult

                    val instanceLoss = this.lossFunction.forward(length, prediction, target)

                    val backwardInstanceLoss = this.lossFunction.backward(prediction, target)

                    this.backwardPropagator.backward(withinBatch, backwardInstanceLoss)

                    batchLoss += instanceLoss

                }

                iterationLoss += batchLoss

                for (optimizable in this.optimizables) {
                    optimizable.optimize(batch.size)
                }

            }

            this.afterEachIteration?.invoke(indexIteration, iterationLoss)
        }

        val stop = System.currentTimeMillis()

        val totalTime = stop - start

        val propagationTimes = this.forwardPropagator.stopTimer() to this.backwardPropagator.stopTimer()

        return totalTime to propagationTimes
    }

}