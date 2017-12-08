package com.komputation.cpu.functions.activation

import com.komputation.cpu.functions.hadamard
import com.komputation.matrix.FloatMath

fun exponentiate(input: FloatArray, result : FloatArray, numberEntries : Int) {
    for(index in 0 until numberEntries) {
        result[index] = FloatMath.exp(input[index])
    }
}

fun backwardExponentiation(forwardEntries: FloatArray, chain : FloatArray, result : FloatArray, numberEntries : Int) {
    hadamard(forwardEntries, chain, result, numberEntries)
}