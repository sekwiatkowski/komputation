package shape.komputation.cuda

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver.*

fun loadKernel(path : String, function: CUfunction, functionName : String) {

    val module = CUmodule()
    cuModuleLoad(module, path)

    cuModuleGetFunction(function, module, functionName)

}

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