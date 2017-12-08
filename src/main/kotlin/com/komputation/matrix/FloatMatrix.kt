package com.komputation.matrix

fun floatMatrix(numberRows : Int, numberColumns : Int, vararg entries : Float) =
    FloatMatrix(entries, numberRows, numberColumns)

fun floatArrayFromRows(numberRows : Int, numberColumns: Int, vararg rows: FloatArray): FloatArray {
    val entries = FloatArray(numberRows * numberColumns)

    for (indexRow in 0 until numberRows) {
        val row = rows[indexRow]

        for (indexColumn in 0 until numberColumns) {
            entries[indexRow + indexColumn * numberRows] = row[indexColumn]
        }
    }

    return entries
}

fun floatArrayFromColumns(numberRows : Int, numberColumns: Int, vararg columns: FloatArray): FloatArray {
    val entries = FloatArray(numberRows * numberColumns)

    for (indexColumn in 0 until numberColumns) {
        val column = columns[indexColumn]

        for (indexRow in 0 until numberRows) {
            entries[indexRow + indexColumn * numberRows] = column[indexRow]
        }
    }

    return entries
}

fun floatMatrixFromRows(numberRows : Int, numberColumns: Int, vararg rows: FloatArray) =
    FloatMatrix(floatArrayFromRows(numberRows, numberColumns, *rows), numberRows, numberColumns)

fun floatMatrixFromColumns(numberRows : Int, numberColumns: Int, vararg rows: FloatArray) =
    FloatMatrix(floatArrayFromColumns(numberRows, numberColumns, *rows), numberRows, numberColumns)

fun oneHotArray(size: Int, index: Int, value : Float = 1.0f): FloatArray {
    val array = FloatArray(size)
    array[index] = value

    return array
}