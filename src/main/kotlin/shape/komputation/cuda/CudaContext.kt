package shape.komputation.cuda

import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.JCudaDriver.cuCtxCreate
import jcuda.driver.JCudaDriver.cuInit

fun initializeCuda() {

    cuInit(0)

}

fun createCudaContext(device : CUdevice): CUcontext {

    val context = CUcontext()
    cuCtxCreate(context, 0, device)

    return context

}
