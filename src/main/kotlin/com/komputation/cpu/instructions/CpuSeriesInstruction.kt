package com.komputation.cpu.instructions

import com.komputation.cpu.layers.recurrent.series.CpuSeries
import com.komputation.instructions.ContinuationInstruction

interface CpuSeriesInstruction : ContinuationInstruction {

    fun buildForCpu() : CpuSeries

}