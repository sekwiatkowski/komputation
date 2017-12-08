package com.komputation.matrix

sealed class Matrix {
    abstract val numberEntries : Int
}

open class FloatMatrix(val entries: FloatArray, val numberRows : Int, val numberColumns : Int) : Matrix() {
    override val numberEntries = entries.size
}

class IntMatrix(val entries : IntArray) : Matrix() {
    override val numberEntries = entries.size
}
