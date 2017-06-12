package shape.komputation.functions

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

fun concatRows(vararg matrices : RealMatrix): RealMatrix {

    val totalNumberColumns = matrices.first().numberColumns()

    var totalNumberRows = 0
    for (matrix in matrices) {
        totalNumberRows += matrix.numberRows()
    }

    val concatenation = createRealMatrix(totalNumberRows, totalNumberColumns)

    var indexConcatenationRow = 0
    for (indexMatrix in 0..matrices.size-1) {

        val matrix = matrices[indexMatrix]

        val numberRows = matrix.numberRows()

        for (indexMatrixRow in 0..numberRows - 1) {

            for (indexMatrixColumn in 0..totalNumberColumns - 1) {

                concatenation.set(indexConcatenationRow, indexMatrixColumn, matrix.get(indexMatrixRow, indexMatrixColumn))

            }

            indexConcatenationRow++

        }

    }

    return concatenation

}

