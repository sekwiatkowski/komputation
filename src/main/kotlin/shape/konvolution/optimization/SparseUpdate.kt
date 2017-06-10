package shape.konvolution.optimization

import shape.konvolution.matrix.RealMatrix

fun updateSparsely(rows: Array<DoubleArray>, rowIndices: IntArray, gradient: RealMatrix, rule : UpdateRule) {

    for (indexRow in 0..gradient.numberRows() - 1) {

        val rowIndex = rowIndices[indexRow]
        val row = rows[rowIndex]

        for (indexColumn in 0..gradient.numberColumns() - 1) {

            val current = row[indexColumn]
            val derivative = gradient.get(indexRow, indexColumn)

            val updated = rule(rowIndex, indexColumn, current, derivative)

            row[indexColumn] = updated

        }

    }

}