package com.komputation.cpu.functions

fun sumRows(numberRows: Int, numberColumns : Int, entries: FloatArray, result : FloatArray) {

    for(indexRow in 0 until numberRows) {

        var sum = 0.0f

        for (indexColumn in 0 until numberColumns) {

            sum += entries[indexColumn * numberRows + indexRow]


        }

        result[indexRow] = sum

    }

}