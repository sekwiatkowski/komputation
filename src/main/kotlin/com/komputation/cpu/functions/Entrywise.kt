package com.komputation.cpu.functions

fun add(a: FloatArray, b: FloatArray, result : FloatArray, numberEntries : Int) {
    for (index in 0 until numberEntries) {
        result[index] = a[index] + b[index]
    }
}

fun hadamard(a: FloatArray, b: FloatArray, result : FloatArray, numberEntries : Int) {
    for (index in 0 until numberEntries) {
        result[index] = a[index] * b[index]
    }
}

fun scale(vector: FloatArray, scalar : Float, result : FloatArray, numberEntries: Int) {
    for (index in 0 until numberEntries) {
        result[index] = scalar * vector[index]
    }
}