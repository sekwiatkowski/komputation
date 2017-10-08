package com.komputation.cuda.kernels

import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver.cuModuleGetFunction
import jcuda.driver.JCudaDriver.cuModuleLoadData
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import java.io.File

fun compileKernel(
    program : nvrtcProgram,
    computeCapabilities : Pair<Int, Int>,
    sourceCode : String,
    name : String,
    nameExpressions : Array<String>,
    headers : Array<String>,
    includeNames : Array<String>) {

    val numberHeaders = headers.size

    nvrtcCreateProgram(program, sourceCode, name, numberHeaders, headers, includeNames)

    for (nameExpression in nameExpressions) {

        JNvrtc.nvrtcAddNameExpression(program, nameExpression)

    }

    val (major, minor) = computeCapabilities

    val options = arrayOf("-arch=compute_$major$minor")
    nvrtcCompileProgram(program, options.size, options)

    val programLogArray = Array(1) { "" }
    nvrtcGetProgramLog(program, programLogArray)

    val programLog = programLogArray.single()

    if (programLog.isNotEmpty()) {

        throw Exception(programLog)

    }

}

fun loadKernel(
    kernel : CUfunction,
    program : nvrtcProgram,
    nameExpression: String) {

    val ptxs = arrayOfNulls<String>(1)
    nvrtcGetPTX(program, ptxs)
    val ptx = ptxs.single()

    val module = CUmodule()
    cuModuleLoadData(module, ptx)

    val loweredName = arrayOfNulls<String>(1)
    nvrtcGetLoweredName(program, nameExpression, loweredName)

    cuModuleGetFunction(kernel, module, loweredName[0])

}