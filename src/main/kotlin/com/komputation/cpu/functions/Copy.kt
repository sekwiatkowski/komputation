package com.komputation.cpu.functions

fun copy(segment : IntArray, start : Int, length : Int, result : IntArray) {
    System.arraycopy(segment, 0, result, start, length)
}

fun copy(segment : FloatArray, start : Int, length : Int, result : FloatArray) {
    System.arraycopy(segment, 0, result, start, length)
}