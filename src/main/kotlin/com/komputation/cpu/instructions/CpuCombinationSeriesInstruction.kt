package com.komputation.cpu.instructions

import com.komputation.cpu.layers.recurrent.series.CpuCombinationSeries
import com.komputation.cpu.layers.recurrent.series.CpuSeries
import com.komputation.instructions.ContinuationInstruction

interface CpuCombinationSeriesInstruction : ContinuationInstruction {

    fun buildForCpu() : CpuCombinationSeries

}