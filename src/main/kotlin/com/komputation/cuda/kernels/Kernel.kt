package com.komputation.cuda.kernels

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram
import jcuda.nvrtc.nvrtcProgram

class Kernel(
    private val computeCapabilities : Pair<Int, Int>,
    private val sourceCode : String,
    private val name : String,
    private val nameExpression : String,
    private val headers : Array<String>,
    private val includeNames : Array<String>) {

    private val program = nvrtcProgram()
    private val kernel = CUfunction()

    init {
        val ptx = compileKernel(this.program, this.computeCapabilities, this.sourceCode, this.name, arrayOf(this.nameExpression), this.headers, this.includeNames)

        loadKernel(this.kernel, ptx, this.program, this.nameExpression)
    }

    fun launch(pointerToParameters: Pointer, numberBlocksInXDimension : Int, numberBlocksInYDimension : Int, numberThreadsPerBlock : Int, sharedMemoryBytes : Int) =
        launchKernel(this.kernel, pointerToParameters, numberBlocksInXDimension, numberBlocksInYDimension, numberThreadsPerBlock, sharedMemoryBytes)

    fun destroy() {
        nvrtcDestroyProgram(this.program)
    }

}