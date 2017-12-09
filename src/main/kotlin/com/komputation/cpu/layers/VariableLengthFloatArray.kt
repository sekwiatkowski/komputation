package com.komputation.cpu.layers

class VariableLengthFloatArray(
    private val numberRows : Int,
    private val minimumLength: Int,
    private val possibleLengths : IntArray,
    private val computeOutputLength : (Int) -> Int) {

    private var store = Array(this.possibleLengths.size) { index ->
        val inputLength = this.possibleLengths[index]
        val outputLength = this.computeOutputLength(inputLength)

        FloatArray(this.numberRows * outputLength)
    }

    fun get(length: Int) =
        this.store[computeLengthIndex(length, this.minimumLength)]

}