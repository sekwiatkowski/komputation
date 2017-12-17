package com.komputation.cpu.layers

interface CpuForwardResult {
    val forwardResult: FloatArray
    val numberOutputColumns: Int
}

interface CpuBackwardResult {
    val backwardResult: FloatArray
    val numberInputColumns: Int
}