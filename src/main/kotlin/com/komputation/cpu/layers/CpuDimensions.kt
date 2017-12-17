package com.komputation.cpu.layers

interface CpuForwardDimensions {
    val numberOutputRows: Int
    val possibleOutputLengths : IntArray
}

interface CpuBackwardDimensions {
    val numberInputRows: Int
    val possibleInputLengths : IntArray
}