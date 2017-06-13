package shape.komputation.functions.activation

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

fun relu(input: RealMatrix) =

    createRealMatrix(input.numberRows(), input.numberColumns()).let { activated ->

        for (indexRow in 0..input.numberRows() - 1) {

            for (indexColumn in 0..input.numberColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                activated.set(indexRow, indexColumn, relu(entry))

            }
        }

        activated
    }

fun relu(entry: Double) =

    Math.max(entry, 0.0)
