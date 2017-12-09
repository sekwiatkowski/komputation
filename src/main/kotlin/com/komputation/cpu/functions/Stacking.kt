package com.komputation.cpu.functions

/*
    A = 1 2
        3 4
    B = 5 6
        7 8
    C = 1 2
        3 4
        5 6
        7 8
 */
fun stackRows(heights : IntArray, totalNumberRows : Int, numberColumns : Int, stacked : FloatArray, vararg arrays: FloatArray) {
    var startAtRow = 0

    for(indexArray in 0 until arrays.size) {
        val numberRows = heights[indexArray]

        val array = arrays[indexArray]

        for (indexColumn in 0 until numberColumns) {
            val startAtIndex = indexColumn * totalNumberRows

            for (indexMatrixRow in 0 until numberRows) {
                stacked[startAtIndex + startAtRow + indexMatrixRow] = array[indexColumn * numberRows + indexMatrixRow]
            }
        }

        startAtRow += numberRows
    }
}