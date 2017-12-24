package com.komputation.instructions.combination

import com.komputation.cpu.instructions.CpuCombinationInstruction
import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.instructions.continuation.BaseEntrywiseInstruction

class Addition(private val name : String?) : BaseEntrywiseInstruction(), CpuCombinationInstruction {

    override fun buildForCpu() =
        CpuAdditionCombination(this.name, this.numberInputRows, 1, this.maximumNumberInputColumns)
}

fun addition(name : String?) = Addition(name)