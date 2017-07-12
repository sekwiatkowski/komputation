package shape.komputation.cuda

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver

fun loadKernel(path : String, function: CUfunction, functionName : String) {

    val module = CUmodule()
    JCudaDriver.cuModuleLoad(module, path)

    JCudaDriver.cuModuleGetFunction(function, module, functionName)

}

fun launchKernel(function : CUfunction, parameters : Pointer, numberBlocks: Int, numberThreadsPerBlock: Int) {

    JCudaDriver.cuLaunchKernel(
        function,
        numberBlocks, 1, 1,
        numberThreadsPerBlock, 1, 1,
        0,
        null,
        parameters,
        null
    )

}