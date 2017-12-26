package com.komputation.cuda.kernels.launch

import com.komputation.matrix.IntMath

fun computeNumberOfThreadsForRows(numberRows : Int, warpSize: Int, maximumNumberThreadsPerBlock : Int) : Pair<Int, Int> {

    if (numberRows < maximumNumberThreadsPerBlock) {

        val numberIterations = 1

        val numberWarps = computeNumberSegments(numberRows, warpSize)
        val numberThreads = numberWarps * warpSize

        return numberIterations to numberThreads

    }
    else {

        val numberIterations = computeNumberSegments(numberRows, maximumNumberThreadsPerBlock)

        val iterativeWarpSize = warpSize * numberIterations
        val numberIterativeWarps = computeNumberSegments (numberRows, iterativeWarpSize)
        val numberThreads = numberIterativeWarps * warpSize

        return numberIterations to numberThreads

    }

}

fun computeEntrywiseLaunchConfiguration(
    numberElements : Int,
    numberMultiProcessors : Int,
    numberResidentWarps : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int): KernelLaunchConfiguration {

    val numberRequiredWarps = computeNumberSegments(numberElements, warpSize)

    // No more than one block per multi-processor
    if (numberRequiredWarps <= numberMultiProcessors) {

        val numberBlocks = numberRequiredWarps

        return KernelLaunchConfiguration(numberBlocks, warpSize, 1)

    }
    // More than one block per multi-processor
    else {

        val numberAvailableWarps = numberResidentWarps * numberMultiProcessors

        val numberOccupiedWarps = IntMath.min(numberRequiredWarps, numberAvailableWarps)
        val numberOccupiedWarpsPerMultiprocessor = computeNumberSegments(numberOccupiedWarps, numberMultiProcessors)

        val numberRequiredThreadsPerMultiprocessor = numberOccupiedWarpsPerMultiprocessor * warpSize
        val numberThreadsPerBlock = Math.min(numberRequiredThreadsPerMultiprocessor, maximumNumberThreadsPerBlock)

        val numberBlocksPerMultiprocessor = computeNumberSegments(numberRequiredThreadsPerMultiprocessor, maximumNumberThreadsPerBlock)

        val numberBlocks = numberMultiProcessors * numberBlocksPerMultiprocessor

        val numberOccupiedThreads = numberBlocks * maximumNumberThreadsPerBlock

        val numberIterations = computeNumberSegments(numberElements, numberOccupiedThreads)

        return KernelLaunchConfiguration(numberBlocks, numberThreadsPerBlock, numberIterations)

    }

}