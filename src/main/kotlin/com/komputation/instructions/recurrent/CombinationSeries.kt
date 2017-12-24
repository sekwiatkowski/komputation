package com.komputation.instructions.recurrent

import com.komputation.cpu.instructions.CpuCombinationInstruction
import com.komputation.cpu.instructions.CpuCombinationSeriesInstruction
import com.komputation.cpu.layers.recurrent.series.CpuCombinationSeries
import com.komputation.instructions.continuation.BaseHigherOrderInstruction

class CombinationSeries internal constructor(private val name : String?, private val steps : Array<CpuCombinationInstruction>) : BaseHigherOrderInstruction(), CpuCombinationSeriesInstruction {

    override fun getLayers() = this.steps

    override fun buildForCpu() =
        CpuCombinationSeries(this.name, Array(this.steps.size) { index -> this.steps[index].buildForCpu() })
}

fun combinationSeries(name : String?, steps: Array<CpuCombinationInstruction>) =
    CombinationSeries(name, steps)
