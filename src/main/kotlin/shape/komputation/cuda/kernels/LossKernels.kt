package shape.komputation.cuda.kernels

object LossKernels {

    fun squaredLoss(blockSize: Int) =
        KernelInstruction(
            "squaredLossKernel",
            "squaredLossKernel<$blockSize>",
            "loss/squaredloss/SquaredLossKernel.cu",
            listOf(KernelHeaders.reduction))

    fun backwardSquaredLoss() =
        KernelInstruction(
            "backwardSquaredLossKernel",
            "backwardSquaredLossKernel",
            "loss/squaredloss/BackwardSquaredLossKernel.cu",
            listOf(KernelHeaders.zero))

    fun logisticLoss(blockSize: Int) =
        KernelInstruction(
            "logisticLossKernel",
            "logisticLossKernel<$blockSize>",
            "loss/logisticloss/LogisticLossKernel.cu",
            listOf(KernelHeaders.reduction))

    fun backwardLogisticLoss() =
        KernelInstruction(
            "backwardLogisticLossKernel",
            "backwardLogisticLossKernel",
            "loss/logisticloss/BackwardLogisticLossKernel.cu",
            listOf(KernelHeaders.zero))

}