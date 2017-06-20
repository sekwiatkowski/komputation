package shape.komputation.matrix

val EMPTY_DOUBLE_MATRIX = DoubleMatrix(0, 0, doubleArrayOf())

fun doubleScalar(value: Double) = DoubleMatrix(1, 1, doubleArrayOf(value))

fun doubleColumnVector(vararg entries : Double) = DoubleMatrix(entries.size, 1, entries)

fun doubleZeroColumnVector(numberRows : Int) = DoubleMatrix(numberRows, 1, DoubleArray(numberRows))

fun doubleRowVector(vararg entries : Double) = DoubleMatrix(1, entries.size, entries)

fun doubleMatrixFromRows(vararg rows: DoubleMatrix): DoubleMatrix {

    val numberRows = rows.size
    val numberColumns = rows[0].numberColumns

    val matrixEntries = DoubleArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        val entries = rows[indexRow].entries

        for (indexColumn in 0..numberColumns - 1) {

            matrixEntries[indexRow + indexColumn * numberRows] = entries[indexColumn]

        }

    }

    return DoubleMatrix(numberRows, numberColumns, matrixEntries)

}

fun oneHotArray(size: Int, index: Int, value : Double = 1.0, otherValue: Double = 0.0): DoubleArray {

    val array = DoubleArray(size) { otherValue }
    array[index] = value

    return array

}

fun oneHotVector(size: Int, index: Int, value : Double = 1.0, otherValue: Double = 0.0) =

    doubleColumnVector(*oneHotArray(size, index, value, otherValue))