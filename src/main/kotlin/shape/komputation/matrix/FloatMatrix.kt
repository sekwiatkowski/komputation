package shape.komputation.matrix

val EMPTY_DOUBLE_MATRIX = FloatMatrix(0, 0, floatArrayOf())

fun floatZeroMatrix(numberRows: Int, numberColumns : Int) = FloatMatrix(numberRows, numberColumns, FloatArray(numberRows * numberColumns))

fun floatRowVector(vararg entries : Float) = FloatMatrix(1, entries.size, entries)

fun floatColumnVector(vararg entries : Float) = FloatMatrix(entries.size, 1, entries)

fun floatConstantColumnVector(numberRows : Int, constant : Float) = FloatMatrix(numberRows, 1, FloatArray(numberRows) { constant })

fun floatZeroColumnVector(numberRows : Int) = floatConstantColumnVector(numberRows, 0.0f)

fun floatScalar(value: Float) = FloatMatrix(1, 1, floatArrayOf(value))

fun floatMatrixFromRows(vararg rows: FloatArray): FloatMatrix {

    val numberRows = rows.size
    val numberColumns = rows[0].size

    val matrixEntries = FloatArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        val entries = rows[indexRow]

        for (indexColumn in 0..numberColumns - 1) {

            matrixEntries[indexRow + indexColumn * numberRows] = entries[indexColumn]

        }

    }

    return FloatMatrix(numberRows, numberColumns, matrixEntries)

}

fun floatMatrixFromColumns(vararg columns: FloatArray): FloatMatrix {

    val numberRows = columns[0].size
    val numberColumns = columns.size
    val entries = FloatArray(numberRows * numberColumns)

    var count = 0

    for (column in columns) {

        for (entry in column) {

            entries[count++] = entry

        }

    }

    return FloatMatrix(numberRows, numberColumns, entries)

}

fun sequence(numberSteps : Int, generateStep : (Int) -> FloatArray): FloatMatrix {

    return floatMatrixFromColumns(*Array(numberSteps, generateStep))

}


fun oneHotArray(size: Int, index: Int, value : Float = 1.0f): FloatArray {

    val array = FloatArray(size)
    array[index] = value

    return array

}

fun oneHotVector(size: Int, index: Int, value : Float = 1.0f) =

    floatColumnVector(*oneHotArray(size, index, value))