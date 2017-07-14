package shape.komputation.loss

import shape.komputation.cpu.loss.CpuLossFunction

interface CpuLossFunctionInstruction {

    fun buildForCpu() : CpuLossFunction

}