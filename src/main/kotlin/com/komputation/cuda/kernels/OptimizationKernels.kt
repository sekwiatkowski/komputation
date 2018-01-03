package com.komputation.cuda.kernels

object OptimizationKernels {

    fun stochasticGradientDescent () = KernelInstruction(
        "stochasticGradientDescentKernel",
        "optimization/StochasticGradientDescentKernel.cu")

    fun momentum () = KernelInstruction(
        "momentumKernel",
        "optimization/historical/MomentumKernel.cu")

    fun nesterov () = KernelInstruction(
        "nesterovKernel",
        "optimization/historical/NesterovKernel.cu")

    fun adagrad () = KernelInstruction(
        "adagradKernel",
        "optimization/adaptive/AdagradKernel.cu")

    fun adadelta () = KernelInstruction(
        "adadeltaKernel",
        "optimization/adaptive/AdadeltaKernel.cu")

    fun rmsprop () = KernelInstruction(
        "rmspropKernel",
        "optimization/adaptive/RmspropKernel.cu")

    fun adam () = KernelInstruction(
        "adamKernel",
        "optimization/adaptive/AdamKernel.cu")

}