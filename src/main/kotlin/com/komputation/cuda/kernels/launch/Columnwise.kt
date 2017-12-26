package com.komputation.cuda.kernels.launch


// One column per block
fun computeColumnwiseLaunchConfiguration(
    numberRows: Int,
    numberColumns: Int,
    maximumNumberThreadsPerBlock: Int) : KernelLaunchConfiguration {

    if (numberRows <= maximumNumberThreadsPerBlock) {

        return KernelLaunchConfiguration(numberColumns, numberRows, 1)

    }
    else {

        val numberIterations = computeNumberSegments(numberRows, maximumNumberThreadsPerBlock)

        return KernelLaunchConfiguration(numberColumns, maximumNumberThreadsPerBlock, numberIterations)

    }

}
