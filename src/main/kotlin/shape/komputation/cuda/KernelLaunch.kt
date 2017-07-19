package shape.komputation.cuda

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver.cuLaunchKernel

fun launchKernel(function : CUfunction, parameters : Pointer, numberBlocks: Int, numberThreadsPerBlock: Int, sharedMemoryBytes : Int) =

    cuLaunchKernel(
        function,
        numberBlocks, 1, 1,
        numberThreadsPerBlock, 1, 1,
        sharedMemoryBytes,
        null,
        parameters,
        null
    )