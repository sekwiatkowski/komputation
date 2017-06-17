package shape.komputation.matrix

sealed class Matrix

data class DoubleMatrix(val numberRows : Int, val numberColumns : Int, val entries : DoubleArray) : Matrix()

val EMPTY_DOUBLE_MATRIX = DoubleMatrix(0, 0, doubleArrayOf())

fun doubleZeroMatrix(numberRows: Int, numberColumns: Int) = DoubleMatrix(numberRows, numberColumns, DoubleArray(numberRows * numberColumns))

fun doubleScalar(value: Double) = DoubleMatrix(1, 1, doubleArrayOf(value))

fun doubleRowVector(vararg entries : Double) = DoubleMatrix(entries.size, 1, entries)

fun doubleZeroRowVector(numberRows : Int) = DoubleMatrix(numberRows, 1, DoubleArray(numberRows))

fun doubleColumnVector(vararg entries : Double) = DoubleMatrix(1, entries.size, entries)

fun doubleRowMatrix(vararg columnVectors: DoubleMatrix): DoubleMatrix {

    val numberRows = columnVectors.size
    val numberColumns = columnVectors[0].numberColumns

    val matrixEntries = DoubleArray(numberRows * numberColumns)

    for (indexRow in 0..numberRows - 1) {

        val entries = columnVectors[indexRow].entries

        for (indexColumn in 0..numberColumns - 1) {

            matrixEntries[indexRow + indexColumn * numberRows] = entries[indexColumn]

        }

    }

    return DoubleMatrix(numberRows, numberColumns, matrixEntries)

}

fun oneHotVector(size: Int, index: Int, value : Double = 1.0): DoubleMatrix {

    val vector = doubleZeroRowVector(size)
    vector.entries[index] = value

    return vector

}

data class IntMatrix(val entries : IntArray, val numberRows : Int, val numberColumns : Int) : Matrix()

fun intVector(vararg entries : Int) = IntMatrix(entries, entries.size, 1)
