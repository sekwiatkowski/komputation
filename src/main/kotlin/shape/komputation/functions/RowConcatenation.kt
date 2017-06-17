package shape.komputation.functions

import shape.komputation.matrix.DoubleMatrix

fun concatRows(vararg matrices : DoubleMatrix): DoubleMatrix {

    var totalNumberRows = 0
    for (matrix in matrices) {
        totalNumberRows += matrix.numberRows
    }

    val numberColumns = matrices.first().numberColumns

    val concatenation = DoubleArray(totalNumberRows * numberColumns)

    var startAtRow = 0
    for(indexMatrix in 0..matrices.size - 1) {

        val matrix = matrices[indexMatrix]
        val matrixEntries = matrix.entries
        val numberMatrixRows = matrix.numberRows

        for (indexMatrixColumn in 0..numberColumns - 1) {

            val startAtIndex = indexMatrixColumn * totalNumberRows

            for (indexMatrixRow in 0..numberMatrixRows - 1) {

                concatenation[startAtIndex + startAtRow + indexMatrixRow] = matrixEntries[indexMatrixColumn * numberMatrixRows + indexMatrixRow]

            }

        }

        startAtRow += numberMatrixRows

    }

    return DoubleMatrix(totalNumberRows, numberColumns, concatenation)

}

