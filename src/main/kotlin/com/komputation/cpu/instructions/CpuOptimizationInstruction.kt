package com.komputation.cpu.instructions

import com.komputation.cpu.optimization.CpuOptimizationStrategy

interface CpuOptimizationInstruction {

    fun buildForCpu() : CpuOptimizationStrategy

}