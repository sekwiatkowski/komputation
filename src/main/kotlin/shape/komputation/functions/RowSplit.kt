package shape.komputation.functions

import shape.komputation.matrix.DoubleMatrix

fun splitRows(matrix : DoubleMatrix, heights : IntArray): Array<DoubleMatrix> {

    val numberColumns = matrix.numberColumns
    val numberRows = matrix.numberRows
    val matrixEntries = matrix.entries

    var runningHeight = 0

    return Array(heights.size) { indexMatrix ->

        val height = heights[indexMatrix]

        val subEntries = DoubleArray(height * numberColumns)

        for (indexColumn in 0..numberColumns - 1) {

            val entriesBeforeSubColumn = indexColumn * height
            val entriesBeforeConcatenationColumn = indexColumn * numberRows + runningHeight

            for (indexSubRow in 0..height - 1) {

                subEntries[entriesBeforeSubColumn + indexSubRow] = matrixEntries[entriesBeforeConcatenationColumn + indexSubRow]

            }

        }

        runningHeight += height

        DoubleMatrix(height, numberColumns, subEntries)

    }

}