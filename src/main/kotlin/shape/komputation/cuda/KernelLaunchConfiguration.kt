package shape.komputation.cuda

import shape.komputation.matrix.IntMath

fun computeKernelLaunchConfigurationForElementWiseFunctions(
    numberElements : Int,
    numberMultiProcessors : Int,
    numberResidentWarps : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int): Triple<Int, Int, Int> {

    val numberRequiredWarps = (numberElements + warpSize - 1) / warpSize

    // No more than one block per multi-processor
    if (numberRequiredWarps <= numberMultiProcessors) {

        val numberBlocks = (numberElements + warpSize - 1) / warpSize

        return Triple(numberBlocks, warpSize, 1)

    }
    // More than one block per multi-processor
    else {

        val numberAvailableWarps = numberResidentWarps * numberMultiProcessors

        val numberOccupiedWarps = IntMath.min(numberRequiredWarps, numberAvailableWarps)
        val numberOccupiedWarpsPerMultiprocessor = (numberOccupiedWarps + numberMultiProcessors - 1) / numberMultiProcessors

        val numberRequiredThreadsPerMultiprocessor = numberOccupiedWarpsPerMultiprocessor * warpSize
        val numberThreadsPerBlock = Math.min(numberRequiredThreadsPerMultiprocessor, maximumNumberThreadsPerBlock)

        val numberBlocksPerMultiprocessor = (numberRequiredThreadsPerMultiprocessor + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        val numberBlocks = numberMultiProcessors * numberBlocksPerMultiprocessor

        val numberOccupiedThreads = numberBlocks * maximumNumberThreadsPerBlock

        val numberIterations = (numberElements + numberOccupiedThreads - 1) / numberOccupiedThreads

        return Triple(numberBlocks, numberThreadsPerBlock, numberIterations)

    }

}