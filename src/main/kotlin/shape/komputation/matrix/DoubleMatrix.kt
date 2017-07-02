package shape.komputation.matrix

val EMPTY_DOUBLE_MATRIX = DoubleMatrix(0, 0, doubleArrayOf())

fun doubleZeroMatrix(numberRows: Int, numberColumns : Int) = DoubleMatrix(numberRows, numberColumns, DoubleArray(numberRows * numberColumns))

fun doubleRowVector(vararg entries : Double) = DoubleMatrix(1, entries.size, entries)

fun doubleColumnVector(vararg entries : Double) = DoubleMatrix(entries.size, 1, entries)

fun doubleConstantColumnVector(numberRows : Int, constant : Double) = DoubleMatrix(numberRows, 1, DoubleArray(numberRows) { constant })

fun doubleZeroColumnVector(numberRows : Int) = doubleConstantColumnVector(numberRows, 0.0)

fun doubleScalar(value: Double) = DoubleMatrix(1, 1, doubleArrayOf(value))

fun doubleMatrixFromRows(vararg rows: DoubleArray): DoubleMatrix {

    val numberRows = rows.size
    val numberColumns = rows[0].size

    val matrixEntries = DoubleArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        val entries = rows[indexRow]

        for (indexColumn in 0..numberColumns - 1) {

            matrixEntries[indexRow + indexColumn * numberRows] = entries[indexColumn]

        }

    }

    return DoubleMatrix(numberRows, numberColumns, matrixEntries)

}

fun doubleMatrixFromColumns(vararg columns: DoubleArray): DoubleMatrix {

    val numberRows = columns[0].size
    val numberColumns = columns.size
    val entries = DoubleArray(numberRows * numberColumns)

    var count = 0

    for (column in columns) {

        for (entry in column) {

            entries[count++] = entry

        }

    }

    return DoubleMatrix(numberRows, numberColumns, entries)

}

fun sequence(numberSteps : Int, generateStep : (Int) -> DoubleArray): DoubleMatrix {

    return doubleMatrixFromColumns(*Array(numberSteps, generateStep))

}


fun oneHotArray(size: Int, index: Int, value : Double = 1.0): DoubleArray {

    val array = DoubleArray(size)
    array[index] = value

    return array

}

fun oneHotVector(size: Int, index: Int, value : Double = 1.0) =

    doubleColumnVector(*oneHotArray(size, index, value))