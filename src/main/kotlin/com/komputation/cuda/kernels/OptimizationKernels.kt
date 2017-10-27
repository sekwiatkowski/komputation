package com.komputation.cuda.kernels

object OptimizationKernels {

    fun stochasticGradientDescent () = KernelInstruction(
        "stochasticGradientDescentKernel",
        "stochasticGradientDescentKernel",
        "optimization/StochasticGradientDescentKernel.cu")

    fun momentum () = KernelInstruction(
        "momentumKernel",
        "momentumKernel",
        "optimization/historical/MomentumKernel.cu")

    fun nesterov () = KernelInstruction(
        "nesterovKernel",
        "nesterovKernel",
        "optimization/historical/NesterovKernel.cu")

    fun adagrad () = KernelInstruction(
        "adagradKernel",
        "adagradKernel",
        "optimization/adaptive/AdagradKernel.cu")

    fun adadelta () = KernelInstruction(
        "adadeltaKernel",
        "adadeltaKernel",
        "optimization/adaptive/AdadeltaKernel.cu")

    fun rmsprop () = KernelInstruction(
        "rmspropKernel",
        "rmspropKernel",
        "optimization/adaptive/rmspropKernel.cu")

}