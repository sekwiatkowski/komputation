package com.komputation.cpu.layers

import com.komputation.matrix.Matrix

interface CpuEntryPoint : CpuLayer, CpuForwardResult {

    fun forward(input: Matrix) : FloatArray

    fun backward(chain : FloatArray) : FloatArray

}