package shape.komputation.cuda.kernels.launch

// One column per block
fun computeRowwiseLaunchConfiguration(
    numberRows : Int,
    numberColumns : Int,
    maximumNumberThreadsPerBlock : Int) : KernelLaunchConfiguration {

    if (numberColumns <= maximumNumberThreadsPerBlock) {

        return KernelLaunchConfiguration(numberRows, numberColumns, 1)

    }
    else {

        val numberIterations = (numberRows + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        return KernelLaunchConfiguration(numberRows, maximumNumberThreadsPerBlock, numberIterations)

    }

}
