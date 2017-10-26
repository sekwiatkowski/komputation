package com.komputation.cuda.kernels.launch

fun computeRowwiseLaunchConfiguration(
    numberRows : Int,
    numberColumns : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int) =

    if (numberColumns <= maximumNumberThreadsPerBlock) {

        val numberWarps = (numberColumns + warpSize - 1) / warpSize
        val numberThreadsPerBlock = numberWarps * warpSize

        KernelLaunchConfiguration(numberRows, numberThreadsPerBlock, 1)

    }
    else {

        val numberIterations = (numberColumns + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        KernelLaunchConfiguration(numberRows, maximumNumberThreadsPerBlock, numberIterations)

    }