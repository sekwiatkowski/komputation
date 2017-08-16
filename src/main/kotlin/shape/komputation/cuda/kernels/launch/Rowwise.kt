package shape.komputation.cuda.kernels.launch

import shape.komputation.matrix.IntMath

// One column per block
fun computeRowwiseLaunchConfiguration(
    numberRows : Int,
    numberColumns : Int,
    maximumNumberThreadsPerBlock : Int) : KernelLaunchConfiguration {

    val numberRequiredThreads = IntMath.closestPowerOfTwo(numberColumns)

    if (numberRequiredThreads <= maximumNumberThreadsPerBlock) {

        return KernelLaunchConfiguration(numberRows, numberRequiredThreads, 1)

    }
    else {

        val numberIterations = (numberRows + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        return KernelLaunchConfiguration(numberRows, maximumNumberThreadsPerBlock, numberIterations)

    }

}
