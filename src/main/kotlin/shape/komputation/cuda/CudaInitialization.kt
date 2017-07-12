package shape.komputation.cuda

import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.JCudaDriver.*

fun initializeCuda() {

    cuInit(0)
    val device = CUdevice()
    cuDeviceGet(device, 0)

    val context = CUcontext()
    cuCtxCreate(context, 0, device)

}