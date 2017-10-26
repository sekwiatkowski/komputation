package com.komputation.cuda.kernels

object LossKernels {

    fun squaredLoss() =
        KernelInstruction(
            "squaredLossKernel",
            "squaredLossKernel",
            "loss/squaredloss/SquaredLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardSquaredLoss() =
        KernelInstruction(
            "backwardSquaredLossKernel",
            "backwardSquaredLossKernel",
            "loss/squaredloss/BackwardSquaredLossKernel.cu",
            listOf(KernelHeaders.nan))

    fun logisticLoss() =
        KernelInstruction(
            "logisticLossKernel",
            "logisticLossKernel",
            "loss/logisticloss/LogisticLossKernel.cu",
            listOf(KernelHeaders.productReduction))

    fun backwardLogisticLoss() =
        KernelInstruction(
            "backwardLogisticLossKernel",
            "backwardLogisticLossKernel",
            "loss/logisticloss/BackwardLogisticLossKernel.cu",
            listOf(KernelHeaders.nan))

    fun crossEntropyLoss() =
        KernelInstruction(
            "crossEntropyLossKernel",
            "crossEntropyLossKernel",
            "loss/crossentropy/CrossEntropyLossKernel.cu",
            listOf(KernelHeaders.sumReduction))

    fun backwardCrossEntropyLoss() =
        KernelInstruction(
            "backwardCrossEntropyLossKernel",
            "backwardCrossEntropyLossKernel",
            "loss/crossentropy/BackwardCrossEntropyLossKernel.cu",
            listOf(KernelHeaders.nan))

}