package shape.komputation.cuda.kernels

import shape.komputation.cuda.computeDeviceFloatArraySize
import shape.komputation.matrix.IntMath

data class EntrywiseLaunchConfiguration(
    val numberBlocks : Int,
    val numberThreadsPerBlock : Int,
    val numberIterations : Int)

fun computeEntrywiseLaunchConfiguration(
    numberElements : Int,
    numberMultiProcessors : Int,
    numberResidentWarps : Int,
    warpSize : Int,
    maximumNumberThreadsPerBlock : Int): EntrywiseLaunchConfiguration {

    val numberRequiredWarps = (numberElements + warpSize - 1) / warpSize

    // No more than one block per multi-processor
    if (numberRequiredWarps <= numberMultiProcessors) {

        val numberBlocks = (numberElements + warpSize - 1) / warpSize

        return EntrywiseLaunchConfiguration(numberBlocks, warpSize, 1)

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

        return EntrywiseLaunchConfiguration(numberBlocks, numberThreadsPerBlock, numberIterations)

    }

}

data class ColumnwiseLaunchConfiguration(
    val numberBlocks : Int,
    val numberThreadsPerBlock : Int,
    val numberIterations : Int,
    val sharedMemoryBytes : Int)


// One column per block
fun computeColumnwiseLaunchConfiguration(
    numberColumns : Int,
    columnSize : Int,
    maximumNumberThreadsPerBlock : Int) : ColumnwiseLaunchConfiguration {

    val numberRequiredThreads = Math.pow(2.0, Math.ceil(Math.log(columnSize.toDouble()) / Math.log(2.0))).toInt()

    if (numberRequiredThreads <= maximumNumberThreadsPerBlock) {

        val sharedMemoryBytes = computeDeviceFloatArraySize(numberRequiredThreads).toInt()

        return ColumnwiseLaunchConfiguration(numberColumns, numberRequiredThreads, 1, sharedMemoryBytes)

    }
    else {

        val numberIterations = (columnSize + maximumNumberThreadsPerBlock - 1) / maximumNumberThreadsPerBlock

        val sharedMemoryBytes = computeDeviceFloatArraySize(maximumNumberThreadsPerBlock).toInt()

        return ColumnwiseLaunchConfiguration(numberColumns, maximumNumberThreadsPerBlock, numberIterations, sharedMemoryBytes)

    }

}
