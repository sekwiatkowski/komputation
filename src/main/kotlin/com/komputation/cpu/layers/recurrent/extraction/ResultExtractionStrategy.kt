package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.layers.CpuForwardResult
import com.komputation.cpu.layers.CpuForwardState

interface ResultExtractionStrategy : CpuForwardState {

    fun extractResult(numberInputColumns : Int): FloatArray

    fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?): FloatArray

}