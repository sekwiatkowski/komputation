package com.komputation.cpu.layers

interface CpuForwardState {
    val forwardResult: FloatArray
    val numberOutputRows: Int
    val numberOutputColumns: Int

}

interface CpuVariableLengthForwardState : CpuForwardState {
    val possibleOutputLengths : IntArray
}

interface CpuBackwardState {
    val backwardResult: FloatArray
    val numberInputRows: Int
    val numberInputColumns: Int

}

interface CpuVariableLengthBackwardState : CpuBackwardState {
    val possibleInputLengths : IntArray
}