package com.komputation.cpu.functions.activation

import com.komputation.matrix.FloatMath

fun tanh(input: FloatArray, result : FloatArray, numberEntries : Int) {
    for (index in 0 until numberEntries) {
        result[index] = FloatMath.tanh(input[index])
    }
}

fun differentiateTanh(input: FloatArray, result : FloatArray, numberEntries : Int) {
    for (index in 0 until numberEntries) {
        val inputEntry = input[index]
        result[index] = 1.0f - (inputEntry * inputEntry)
    }
}