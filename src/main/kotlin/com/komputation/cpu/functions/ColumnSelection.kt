package com.komputation.cpu.functions

fun getColumn(input: FloatArray, indexColumn : Int, rows : Int, result: FloatArray) {
    for (indexRow in 0 until rows) {
        result[indexRow] = input[indexColumn * rows + indexRow]
    }
}

fun setColumn(column : FloatArray, indexColumn : Int, rows : Int, result : FloatArray) {
    for (indexRow in 0 until rows) {
        result[indexColumn * rows + indexRow] = column[indexRow]
    }
}