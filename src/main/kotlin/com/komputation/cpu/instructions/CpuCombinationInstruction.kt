package com.komputation.cpu.instructions

import com.komputation.cpu.layers.CpuCombination
import com.komputation.instructions.ContinuationInstruction

interface CpuCombinationInstruction : ContinuationInstruction {
    fun buildForCpu() : CpuCombination
}