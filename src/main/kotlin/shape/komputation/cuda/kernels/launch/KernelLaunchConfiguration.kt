package shape.komputation.cuda.kernels.launch

data class KernelLaunchConfiguration(
    val numberBlocks : Int,
    val numberThreadsPerBlock : Int,
    val numberIterations : Int)

fun emptyKernelLaunchConfiguration() = KernelLaunchConfiguration(0, 0, 0)