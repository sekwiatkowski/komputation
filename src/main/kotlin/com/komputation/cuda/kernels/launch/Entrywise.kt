package com.komputation.cuda.kernels.launch

import com.komputation.matrix.IntMath

fun computeNumberOfThreadsForRows(numberEntries: Int, warpSize: Int, maximumNumberThreadsPerBlock : Int) : Triple<Int, Int, Int> {
    if (numberEntries < maximumNumberThreadsPerBlock) {
        val numberIterations = 1

        val numberWarps = computeNumberSegments(numberEntries, warpSize)
        val numberThreads = numberWarps * warpSize

        return Triple(numberIterations, numberThreads, numberWarps)

    }
    else {
        val numberIterations = computeNumberSegments(numberEntries, maximumNumberThreadsPerBlock)

        val iterativeWarpSize = warpSize * numberIterations
        val numberIterativeWarps = computeNumberSegments (numberEntries, iterativeWarpSize)
        val numberThreads = numberIterativeWarps * warpSize

        return Triple(numberIterations, numberThreads, numberIterativeWarps)
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