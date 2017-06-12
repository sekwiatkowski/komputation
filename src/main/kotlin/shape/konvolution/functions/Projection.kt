package shape.konvolution.functions

import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

fun project(input: RealMatrix, weights: RealMatrix, bias : RealMatrix?) =

    if (bias != null) {

        if (input.numberColumns() == 1) {

            weights.multiplyAdd(input, bias.copy())
        }
        else {
            weights.multiplyAdd(input, expandBias(bias, input.numberColumns()))
        }

    }
    else {

        weights.multiply(input)
    }

fun expandBias(bias: RealMatrix, inputColumns: Int): RealMatrix {

    val biasRows = bias.numberRows()

    val expandedBiasMatrix = createRealMatrix(biasRows, inputColumns)

    for (indexRow in 0..biasRows - 1) {

        val biasColumns = bias.numberColumns()

        for (indexColumn in 0..biasColumns - 1) {

            expandedBiasMatrix.set(indexRow, indexColumn, bias.get(indexRow, 0))

        }
    }

    return expandedBiasMatrix

}