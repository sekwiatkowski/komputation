package com.komputation.cpu.instructions

import com.komputation.cpu.layers.CpuEntryPoint
import com.komputation.instructions.EntryPointInstruction
import com.komputation.instructions.Instruction

interface CpuEntryPointInstruction : EntryPointInstruction {

    fun buildForCpu() : CpuEntryPoint

}