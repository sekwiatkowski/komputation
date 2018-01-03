package com.komputation.cuda.kernels

object LossKernels {

    fun squaredLoss() =
        KernelInstruction(
            "squaredLossKernel",
            "loss/squaredloss/SquaredLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardSquaredLoss() =
        KernelInstruction(
            "backwardSquaredLossKernel",
            "loss/squaredloss/BackwardSquaredLossKernel.cu",
            listOf(KernelHeaders.nan))

    fun logisticLoss() =
        KernelInstruction(
            "logisticLossKernel",
            "loss/logisticloss/LogisticLossKernel.cu",
            listOf(KernelHeaders.productReduction))

    fun backwardLogisticLoss() =
        KernelInstruction(
            "backwardLogisticLossKernel",
            "loss/logisticloss/BackwardLogisticLossKernel.cu",
            listOf(KernelHeaders.nan))

    fun crossEntropyLoss() =
        KernelInstruction(
            "crossEntropyLossKernel",
            "loss/crossentropy/CrossEntropyLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardCrossEntropyLoss() =
        KernelInstruction(
            "backwardCrossEntropyLossKernel",
            "loss/crossentropy/BackwardCrossEntropyLossKernel.cu",
            listOf(KernelHeaders.nan))

}