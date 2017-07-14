package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss

class LogisticLoss : CpuLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

}