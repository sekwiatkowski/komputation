package shape.komputation.cuda

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram
import jcuda.nvrtc.nvrtcProgram
import shape.komputation.layers.Resourceful
import java.io.File

class Kernel(
    private val computeCapabilities : Pair<Int, Int>,
    private val cuFile : File,
    private val name : String,
    private val nameExpression : String,
    private val headerFiles : Array<File>,
    private val includeNames : Array<String>) : Resourceful {

    private val program = nvrtcProgram()
    private val kernel = CUfunction()

    override fun acquire() {

        compileKernel(this.program, this.computeCapabilities, this.cuFile, this.name, arrayOf(this.nameExpression), this.headerFiles, this.includeNames)

        loadKernel(this.kernel, this.program, this.nameExpression)

    }

    fun launch(pointerToParameters: Pointer, numberBlocks : Int, numberThreadsPerBlock : Int, sharedMemoryBytes : Int) =

        launchKernel(this.kernel, pointerToParameters, numberBlocks, numberThreadsPerBlock, sharedMemoryBytes)

    override fun release() {

        nvrtcDestroyProgram(this.program)

    }

}