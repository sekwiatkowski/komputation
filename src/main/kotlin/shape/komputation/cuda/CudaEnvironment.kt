package shape.komputation.cuda

import jcuda.driver.CUcontext

data class CudaEnvironment(
    val context : CUcontext,
    val computeCapabilities : Pair<Int, Int>,
    val numberThreadsPerBlock : Int,
    val numberBlocks : Int)

fun setUpCudaEnvironment(deviceId : Int = 0): CudaEnvironment {

    initializeCuda()

    val device = getCudaDevice(deviceId)

    val context = createCudaContext(device)

    val computeCapability = queryComputeCapability(device)

    val maximumNumberOfThreads = queryMaximumNumberOfThreadsPerBlock(device)
    val maximumNumberOfBlocks = queryMaximumNumberOfBlocks(device)

    return CudaEnvironment(context, computeCapability, maximumNumberOfThreads, maximumNumberOfBlocks)

}