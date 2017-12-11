package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.layers.CpuVariableLengthForwardState

interface ResultExtractionStrategy : CpuVariableLengthForwardState {

    fun extractResult(numberInputColumns : Int): FloatArray

    fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?): FloatArray

}