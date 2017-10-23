package com.komputation.cpu.functions

fun repeatColumn(column: FloatArray, numberRepetitions: Int, repetition: FloatArray) {

    val numberRows = column.size

    for(indexColumn in 0 until numberRepetitions) {

        val start = indexColumn * numberRows

        for (indexRow in 0 until numberRows) {

            repetition[start+indexRow] = column[indexRow]

        }

    }

}