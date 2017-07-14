package shape.komputation.loss

import shape.komputation.cpu.loss.CpuLogisticLoss

class LogisticLoss : CpuLossFunctionInstruction {

    override fun buildForCpu() =

        CpuLogisticLoss()

}

fun logisticLoss() =

    LogisticLoss()