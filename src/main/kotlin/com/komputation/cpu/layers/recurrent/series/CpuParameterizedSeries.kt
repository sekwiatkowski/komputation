package com.komputation.cpu.layers.recurrent.series

import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.optimization.Optimizable

class CpuParameterizedSeries internal constructor(
    name : String?,
    steps: Array<CpuContinuation>,
    private val sharedParameter: FloatArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule? = null) : CpuSeries(name, steps), Optimizable {

    private val numberEntries = sharedParameter.size

    fun backwardSeries() {
        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()
    }

    override fun optimize(batchSize : Int) {
        if (this.updateRule != null) {
            updateDensely(this.sharedParameter, this.batchAccumulator.getAccumulation(), this.numberEntries, batchSize, this.updateRule)
        }

        this.batchAccumulator.reset()
    }

}