package com.komputation.matrix

fun floatMatrix(numberRows : Int, numberColumns : Int, vararg entries : Float) =
    FloatMatrix(entries, numberRows, numberColumns)

fun floatArrayFromRows(vararg rows: FloatArray): FloatArray {

    val numberRows = rows.size
    val numberColumns = rows[0].size

    val entries = FloatArray(numberRows * numberColumns)

    for (indexRow in 0 until numberRows) {

        val row = rows[indexRow]

        for (indexColumn in 0 until numberColumns) {

            entries[indexRow + indexColumn * numberRows] = row[indexColumn]

        }

    }

    return entries

}

fun floatMatrixFromRows(numberColumns: Int, vararg rows: FloatArray) =
    FloatMatrix(floatArrayFromRows(*rows), rows.size, numberColumns)

fun oneHotArray(size: Int, index: Int, value : Float = 1.0f): FloatArray {

    val array = FloatArray(size)
    array[index] = value

    return array

}