package com.komputation.cpu.layers
class VariableLengthFloatArray(
    private val numberRows : Int,
    private val possibleLengths : IntArray) {

    private val minimumLength = possibleLengths.min()!!

    private var store = Array(this.possibleLengths.size) { index ->
        FloatArray(this.numberRows * this.possibleLengths[index])
    }

    fun get(length: Int): FloatArray {
        val lengthIndex = computeLengthIndex(length, this.minimumLength)
        return this.store[lengthIndex]
    }

}

