package shape.komputation.cuda

import jcuda.driver.CUfunction
import java.io.File

fun acquireKernel(cuFile : File, kernelName: String, kernel: CUfunction, computeCapabilities : Pair<Int, Int>): File {

    val ptxFile = File.createTempFile(kernelName, ".ptx")
    ptxFile.deleteOnExit()

    val ptxPath = ptxFile.path

    compileKernel(cuFile.path, ptxPath, computeCapabilities)

    loadKernel(ptxPath, kernel, kernelName)

    return ptxFile

}
