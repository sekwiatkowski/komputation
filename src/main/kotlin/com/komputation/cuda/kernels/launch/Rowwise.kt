package com.komputation.cuda.kernels.launch

fun computeRowwiseLaunchConfiguration(
    numberRows : Int,
    numberColumns : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int) =

    if (numberColumns <= maximumNumberThreadsPerBlock) {

        val numberWarps = computeNumberSegments(numberColumns, warpSize)
        val numberThreadsPerBlock = numberWarps * warpSize

        KernelLaunchConfiguration(numberRows, numberThreadsPerBlock, 1)

    }
    else {

        val numberIterations = computeNumberSegments(numberColumns, maximumNumberThreadsPerBlock)

        KernelLaunchConfiguration(numberRows, maximumNumberThreadsPerBlock, numberIterations)

    }