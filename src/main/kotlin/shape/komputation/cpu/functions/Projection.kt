package shape.komputation.cpu.functions


import shape.komputation.matrix.createBlasMatrix
import shape.komputation.matrix.createBlasVector

fun project(input: FloatArray, numberInputRows : Int, numberInputColumns: Int, weights: FloatArray, numberWeightRows : Int, numberWeightColumns : Int) =

    project(input, numberInputRows, numberInputColumns, weights, numberWeightRows, numberWeightColumns, null)

fun project(input: FloatArray, numberInputRows : Int, numberInputColumns: Int, weights: FloatArray, numberWeightRows : Int, numberWeightColumns : Int, bias : FloatArray?) : FloatArray {

    val inputMatrix = createBlasMatrix(numberInputRows, numberInputColumns, input)
    val weightMatrix = createBlasMatrix(numberWeightRows, numberWeightColumns, weights)

    if (bias != null) {

        if (numberInputColumns == 1) {

            val biasVector = createBlasVector(bias)

            return weightMatrix.multiplyAdd(inputMatrix, biasVector).getEntries()

        }
        else {

            val expandedBias = FloatArray(numberWeightRows * numberInputColumns)

            repeatColumn(bias, expandedBias, numberInputColumns)

            return weightMatrix.multiplyAdd(inputMatrix, createBlasMatrix(bias.size, numberInputColumns, expandedBias)).getEntries()

        }

    }
    else {

        return weightMatrix.multiply(inputMatrix).getEntries()

    }

}

fun backwardProjectionWrtInput(
    numberInputRows: Int,
    numberInputColumns : Int,
    numberInputEntries : Int,
    weightEntries : FloatArray,
    numberWeightRows : Int,
    chainEntries: FloatArray,
    numberChainRows : Int): FloatArray {

    val derivatives = FloatArray(numberInputEntries)

    var index = 0

    for (indexInputColumn in 0..numberInputColumns - 1) {

        for (indexInputRow in 0..numberInputRows - 1) {

            var derivative = 0.0f

            for (indexWeightRow in 0..numberWeightRows - 1) {

                val chainEntry = chainEntries[indexWeightRow + indexInputColumn * numberChainRows]
                val weightEntry = weightEntries[indexWeightRow + indexInputRow * numberWeightRows]

                derivative += chainEntry * weightEntry

            }

            derivatives[index++] = derivative

        }

    }

    return derivatives

}

fun backwardProjectionWrtWeights(
    numberWeightEntries : Int,
    numberWeightRows : Int,
    numberWeightColumns: Int,
    inputEntries: FloatArray,
    numberInputRows : Int,
    chainEntries: FloatArray,
    numberChainRows: Int,
    numberChainColumns : Int): FloatArray {

    val derivatives = FloatArray(numberWeightEntries)

    var index = 0

    for (indexWeightColumn in 0..numberWeightColumns - 1) {

        for (indexWeightRow in 0..numberWeightRows - 1) {

            var derivative = 0.0f

            for (indexChainColumn in 0..numberChainColumns - 1) {

                // d loss / d pre1, d loss / d pre2
                // All multiplications on other rows equal to zero
                val chainEntry = chainEntries[indexWeightRow + indexChainColumn * numberChainRows]

                // d pre ij / d wk
                val inputEntry = inputEntries[indexWeightColumn + indexChainColumn * numberInputRows]

                derivative += chainEntry * inputEntry

            }

            derivatives[index++] = derivative

        }
    }

    return derivatives

}

fun backwardProjectionWrtBias(numberBiasRows : Int, chain: FloatArray, numberChainRows: Int, numberChainColumns: Int, result : FloatArray) {

    for (indexRow in 0..numberBiasRows - 1) {

        var derivative = 0.0f

        for (indexChainColumn in 0..numberChainColumns - 1) {

            derivative += chain[indexRow + numberChainRows * indexChainColumn]

        }

        result[indexRow] = derivative

    }


}