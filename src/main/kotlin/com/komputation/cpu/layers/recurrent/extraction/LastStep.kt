package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.layers.recurrent.series.CpuSeries

class LastStep(
    numberRows : Int,
    private val isReversed : Boolean) : ResultExtractionStrategy {

    override var forwardResult = FloatArray(0)
    override val numberOutputRows = numberRows
    override val numberOutputColumns = 1

    override val possibleOutputLengths = intArrayOf(1)

    override fun extractResult(series : CpuSeries, numberInputColumns : Int): FloatArray {
        this.forwardResult = series.getForwardResult(if(this.isReversed) 0 else numberInputColumns-1)

        return this.forwardResult
    }

    override fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?) =
        previousBackwardPreviousHiddenState ?: chain

}