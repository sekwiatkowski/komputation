package com.komputation.cuda.kernels

object LossKernels {

    fun squaredLoss() =
        KernelInstruction(
            "squaredLossKernel",
            "loss/squaredloss/SquaredLossKernel.cu")

    fun backwardSquaredLoss() =
        KernelInstruction(
            "backwardSquaredLossKernel",
            "loss/squaredloss/BackwardSquaredLossKernel.cu")

    fun logisticLoss() =
        KernelInstruction(
            "logisticLossKernel",
            "loss/logisticloss/LogisticLossKernel.cu")

    fun backwardLogisticLoss() =
        KernelInstruction(
            "backwardLogisticLossKernel",
            "loss/logisticloss/BackwardLogisticLossKernel.cu")

    fun crossEntropyLoss() =
        KernelInstruction(
            "crossEntropyLossKernel",
            "loss/crossentropy/CrossEntropyLossKernel.cu")

    fun backwardCrossEntropyLoss() =
        KernelInstruction(
            "backwardCrossEntropyLossKernel",
            "loss/crossentropy/BackwardCrossEntropyLossKernel.cu")

}