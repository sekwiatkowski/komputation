package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.layers.recurrent.series.Series

class LastStep(
    private val series: Series,
    numberRows : Int,
    private val isReversed : Boolean) : ResultExtractionStrategy {

    override var forwardResult = FloatArray(0)
    override val numberOutputRows = numberRows
    override val numberOutputColumns = 1

    override fun extractResult(numberInputColumns : Int): FloatArray {
        this.forwardResult = this.series.getForwardResult(if(this.isReversed) 0 else numberInputColumns-1)

        return this.forwardResult
    }

    override fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?) =
        previousBackwardPreviousHiddenState ?: chain

}