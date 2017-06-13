package shape.komputation.functions.activation

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

fun sigmoid(input: RealMatrix) =

    createRealMatrix(input.numberRows(), input.numberColumns()).let { activated ->

        for (indexRow in 0..input.numberRows() - 1) {

            for (indexColumn in 0..input.numberColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                activated.set(indexRow, indexColumn, sigmoid(entry))

            }
        }

        activated
    }

fun sigmoid(x: Double) =

    1.0 / (1.0 + Math.exp(-x))