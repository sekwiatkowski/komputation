package com.komputation.cpu.instructions

import com.komputation.cpu.layers.CpuContinuation
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.instructions.ContinuationInstruction

interface CpuContinuationInstruction : ContinuationInstruction {
    fun buildForCpu() : CpuContinuation
}

interface CpuShareableContinuationInstruction : ContinuationInstruction {
    fun buildForCpu(parameter : FloatArray, accumulator: DenseAccumulator) : CpuContinuation
}