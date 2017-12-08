package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely

class Series internal constructor(
    private val name : String?,
    private val parameter : FloatArray,
    private val steps: Array<CpuForwardLayer>,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule? = null) {

    private val numberEntries = parameter.size

    fun forwardStep(withinBatch : Int, step : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean) =
        this.steps[step].forward(withinBatch, numberInputColumns, input, isTraining)

    fun backwardStep(withinBatch: Int, step: Int, chain: FloatArray) =
        this.steps[step].backward(withinBatch, chain)

    fun backwardSeries() {
        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()
    }

    fun optimize(batchSize : Int) {
        if (this.updateRule != null) {
            updateDensely(this.parameter, this.batchAccumulator.getAccumulation(), this.numberEntries, batchSize, this.updateRule)
        }

        this.batchAccumulator.reset()
    }

}