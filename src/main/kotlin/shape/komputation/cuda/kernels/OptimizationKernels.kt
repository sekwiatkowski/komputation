package shape.komputation.cuda.kernels

object OptimizationKernels {

    fun stochasticGradientDescent () = KernelInstruction(
        "stochasticGradientDescentKernel",
        "stochasticGradientDescentKernel",
        "optimization/stochasticgradientdescent/StochasticGradientDescentKernel.cu")

    fun momentum () = KernelInstruction(
        "momentumKernel",
        "momentumKernel",
        "optimization/momentum/MomentumKernel.cu")

}