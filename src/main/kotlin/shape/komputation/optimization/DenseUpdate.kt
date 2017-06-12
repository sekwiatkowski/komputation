package shape.komputation.optimization

import shape.komputation.matrix.RealMatrix

fun updateDensely(parameter: RealMatrix, gradient : RealMatrix, rule : UpdateRule) {

    for (indexRow in 0..parameter.numberRows() - 1) {

        for (indexColumn in 0..parameter.numberColumns() - 1) {

            val current = parameter.get(indexRow, indexColumn)
            val derivative = gradient.get(indexRow, indexColumn)

            val updated = rule(indexRow, indexColumn, current, derivative)

            parameter.set(indexRow, indexColumn, updated)

        }

    }

}