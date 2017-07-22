package shape.komputation.cpu.functions

import shape.komputation.matrix.FloatMatrix

fun splitRows(matrix : FloatMatrix, heights : IntArray): Array<FloatMatrix> {

    val numberColumns = matrix.numberColumns
    val numberRows = matrix.numberRows
    val matrixEntries = matrix.entries

    var runningHeight = 0

    return Array(heights.size) { indexMatrix ->

        val height = heights[indexMatrix]

        val subEntries = FloatArray(height * numberColumns)

        for (indexColumn in 0..numberColumns - 1) {

            val entriesBeforeSubColumn = indexColumn * height
            val entriesBeforeConcatenationColumn = indexColumn * numberRows + runningHeight

            System.arraycopy(matrixEntries, entriesBeforeConcatenationColumn, subEntries, entriesBeforeSubColumn, height)

        }

        runningHeight += height

        FloatMatrix(height, numberColumns, subEntries)

    }

}