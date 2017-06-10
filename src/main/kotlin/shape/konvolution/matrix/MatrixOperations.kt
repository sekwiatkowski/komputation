package shape.konvolution.matrix


fun concatColumns(matrices : Array<RealMatrix>, numberMatrices : Int, numberRows : Int, numberColumns : Int): RealMatrix {

    val concatenation = createRealMatrix(numberRows, numberColumns)

    var indexColumnConcatenation = 0

    for (indexMatrix in 0..numberMatrices - 1) {

        val matrix = matrices[indexMatrix]
        val numberColumnsMatrix = matrix.numberColumns()

        for (indexColumnMatrix in 0..numberColumnsMatrix - 1) {

            for (indexRow in 0..numberRows - 1) {

                concatenation.set(indexRow, indexColumnConcatenation, matrix.get(indexRow, indexColumnMatrix))

            }

            indexColumnConcatenation++

        }

    }

    return concatenation

}

fun softmax(input: RealMatrix) =

    createRealMatrix(input.numberRows(), input.numberColumns()).let { activated ->

        for (indexColumn in 0..input.numberColumns() - 1) {

            val exponentiated = Array(input.numberRows()) { indexRow ->

                Math.exp(input.get(indexRow, indexColumn))

            }

            val sum = exponentiated.sum()

            for (indexRow in 0..input.numberRows() - 1) {

                val activation = exponentiated[indexRow].div(sum)

                activated.set(indexRow, indexColumn, activation)
            }


        }

        activated

    }

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