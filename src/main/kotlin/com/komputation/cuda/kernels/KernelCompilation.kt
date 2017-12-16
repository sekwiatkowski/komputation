package com.komputation.cuda.kernels

import jcuda.Pointer
import jcuda.driver.*
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import java.nio.ByteBuffer

fun compileKernel(
    program : nvrtcProgram,
    computeCapabilities : Pair<Int, Int>,
    sourceCode : String,
    name : String,
    nameExpressions : Array<String>,
    headers : Array<String>,
    includeNames : Array<String>) : String {

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

    val ptxArray = Array(1) { "" }
    nvrtcGetPTX(program, ptxArray)
    val ptx = ptxArray.single()

    return ptx
}

fun link(linkState : CUlinkState, ptx : String, name : String, devrtPath : String) : Pointer {
    cuLinkCreate(JITOptions(), linkState)

    cuLinkAddFile(linkState, CUjitInputType.CU_JIT_INPUT_LIBRARY, devrtPath, JITOptions())

    val bytes = ptx.toByteArray()
    val byteBuffer = ByteBuffer.wrap(bytes)
    cuLinkAddData(linkState, CUjitInputType.CU_JIT_INPUT_PTX, Pointer.to(byteBuffer), bytes.size.toLong(), name, JITOptions())

    val cubinPointer = Pointer()
    val linkSize = LongArray(1)
    cuLinkComplete(linkState, cubinPointer, linkSize)

    return cubinPointer
}

fun loadKernel(kernel : CUfunction, ptx : String, program : nvrtcProgram, nameExpression: String) {
    val module = CUmodule()
    cuModuleLoadData(module, ptx)

    val loweredName = arrayOfNulls<String>(1)
    nvrtcGetLoweredName(program, nameExpression, loweredName)

    cuModuleGetFunction(kernel, module, loweredName[0])
}

fun loadKernel(kernel : CUfunction, cubinPointer : Pointer, name: String) {
    // https://stackoverflow.com/questions/32535828/jit-in-jcuda-loading-multiple-ptx-modules
    val module = CUmodule()
    cuModuleLoadDataEx(module, cubinPointer, 0, IntArray(0), Pointer.to(IntArray(0)))

    cuModuleGetFunction(kernel, module, name)
}
