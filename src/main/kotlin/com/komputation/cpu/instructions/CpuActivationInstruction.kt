package com.komputation.cpu.instructions

import com.komputation.cpu.layers.continuation.activation.CpuActivation

interface CpuActivationInstruction : CpuContinuationInstruction {
    override fun buildForCpu(): CpuActivation
}