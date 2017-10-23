package com.komputation.cpu.functions

fun transpose(numberRows: Int, numberColumns: Int, entries: FloatArray, result : FloatArray) {

    if(numberRows == 1 || numberColumns == 1) {

        System.arraycopy(entries, 0, result, 0, entries.size)

        return

    }

    for (indexColumn in 0 until numberColumns) {

        val start = indexColumn * numberRows

        for(indexRow in 0 until numberRows) {

            result[indexRow * numberColumns + indexColumn] = entries[start + indexRow]

        }

    }

}