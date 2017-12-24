package com.komputation.cpu.instructions

import com.komputation.cpu.loss.CpuLossFunction
import com.komputation.instructions.LossInstruction

interface CpuLossFunctionInstruction : LossInstruction {

    fun buildForCpu() : CpuLossFunction

}