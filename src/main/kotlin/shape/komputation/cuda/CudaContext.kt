package shape.komputation.cuda

import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.JCudaDriver.*

data class CudaContext(
    val context : CUcontext,
    val computeCapabilities : Pair<Int, Int>,
    val maximumNumberThreadsPerBlock: Int,
    val numberBlocks : Int) {

    val kernelFactory = KernelFactory(this.computeCapabilities)

    fun destroy() {

        destroyCudaContext(this.context)

    }

}

fun setUpCudaContext(deviceId : Int = 0): CudaContext {

    initializeCuda()

    val device = getCudaDevice(deviceId)

    val context = createCudaContext(device)

    val computeCapability = queryComputeCapability(device)

    val maximumNumberOfThreads = queryMaximumNumberOfThreadsPerBlock(device)
    val maximumNumberOfBlocks = queryMaximumNumberOfBlocks(device)

    return CudaContext(context, computeCapability, maximumNumberOfThreads, maximumNumberOfBlocks)

}

fun initializeCuda() {

    cuInit(0)

}

fun createCudaContext(device : CUdevice): CUcontext {

    val context = CUcontext()
    cuCtxCreate(context, 0, device)

    return context

}

fun destroyCudaContext(context : CUcontext) =

    cuCtxDestroy(context)