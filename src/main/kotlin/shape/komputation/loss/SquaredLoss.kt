package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss

class SquaredLoss : CpuLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

}