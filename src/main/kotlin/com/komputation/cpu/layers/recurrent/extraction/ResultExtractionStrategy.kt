package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.layers.CpuForwardState
import com.komputation.cpu.layers.recurrent.series.CpuSeries

interface ResultExtractionStrategy : CpuForwardState {

    fun extractResult(series : CpuSeries, numberInputColumns : Int): FloatArray

    fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?): FloatArray

}