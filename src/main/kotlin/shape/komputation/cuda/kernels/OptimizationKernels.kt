package shape.komputation.cuda.kernels

object OptimizationKernels {

    fun stochasticGradientDescent () = KernelInstruction(
        "stochasticGradientDescentKernel",
        "stochasticGradientDescentKernel",
        "optimization/stochasticgradientdescent/StochasticGradientDescentKernel.cu")

}