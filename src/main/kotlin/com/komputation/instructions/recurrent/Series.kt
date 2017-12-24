package com.komputation.instructions.recurrent

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.instructions.CpuSeriesInstruction
import com.komputation.cpu.layers.recurrent.series.CpuSeries
import com.komputation.instructions.continuation.BaseHigherOrderInstruction

class Series internal constructor(private val name : String?, private val steps : Array<CpuContinuationInstruction>) : BaseHigherOrderInstruction(), CpuSeriesInstruction {

    override fun getLayers() = this.steps

    override fun buildForCpu() =
        CpuSeries(this.name, Array(this.steps.size) { index -> this.steps[index].buildForCpu() })
}

fun series(name : String?, steps: Array<CpuContinuationInstruction>) =
    Series(name, steps)