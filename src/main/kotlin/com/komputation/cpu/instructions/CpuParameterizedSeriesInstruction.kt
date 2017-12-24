package com.komputation.cpu.instructions

import com.komputation.cpu.layers.recurrent.series.CpuParameterizedSeries
import com.komputation.cpu.layers.recurrent.series.CpuSeries
import com.komputation.instructions.ContinuationInstruction

interface CpuParameterizedSeriesInstruction : ContinuationInstruction {

    fun buildForCpu() : CpuParameterizedSeries

}