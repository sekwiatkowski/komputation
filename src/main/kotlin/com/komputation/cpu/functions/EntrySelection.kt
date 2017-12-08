package com.komputation.cpu.functions

fun selectEntries(input: FloatArray, indices : IntArray, result : FloatArray, numberIndices : Int) {
    for (index in 0 until numberIndices) {
        result[index] = input[indices[index]]
    }
}