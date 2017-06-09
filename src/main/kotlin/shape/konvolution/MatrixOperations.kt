package shape.konvolution

import no.uib.cipr.matrix.DenseMatrix
import no.uib.cipr.matrix.Matrix

fun createDenseMatrix(vararg rows: DoubleArray) =

    DenseMatrix(rows)

fun createDenseMatrix(numberRows: Int, numberColumns: Int) =

    DenseMatrix(numberRows, numberColumns)

fun concatColumns(matrices : Array<Matrix>, numberMatrices : Int, numberRows : Int, numberColumns : Int): Matrix {

    val concatenation = createDenseMatrix(numberRows, numberColumns)

    var indexColumnConcatenation = 0

    for (indexMatrix in 0..numberMatrices - 1) {

        val matrix = matrices[indexMatrix]
        val numberColumnsMatrix = matrix.numColumns()

        for (indexColumnMatrix in 0..numberColumnsMatrix - 1) {

            for (indexRow in 0..numberRows - 1) {

                concatenation.set(indexRow, indexColumnConcatenation, matrix.get(indexRow, indexColumnMatrix))

            }

            indexColumnConcatenation++

        }

    }

    return concatenation

}

fun softmax(input: Matrix) =

    DenseMatrix(input.numRows(), input.numColumns()).let { activated ->

        for (indexColumn in 0..input.numColumns() - 1) {

            val exponentiated = Array(input.numRows()) { indexRow ->

                Math.exp(input.get(indexRow, indexColumn))

            }

            val sum = exponentiated.sum()

            for (indexRow in 0..input.numRows() - 1) {

                val activation = exponentiated[indexRow].div(sum)

                activated.set(indexRow, indexColumn, activation)
            }


        }

        activated

    }

fun sigmoid(input: Matrix) =

    DenseMatrix(input.numRows(), input.numColumns()).let { activated ->

        for (indexRow in 0..input.numRows() - 1) {

            for (indexColumn in 0..input.numColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                activated.set(indexRow, indexColumn, sigmoid(entry))

            }
        }

        activated
    }

fun sigmoid(x: Double) =

    1.0 / (1.0 + Math.exp(-x))

fun relu(input: Matrix) =

    DenseMatrix(input.numRows(), input.numColumns()).let { activated ->

        for (indexRow in 0..input.numRows() - 1) {

            for (indexColumn in 0..input.numColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                activated.set(indexRow, indexColumn, relu(entry))

            }
        }

        activated
    }

fun relu(entry: Double) =

    Math.max(entry, 0.0)

fun project(input: Matrix, weights: Matrix, bias : Matrix?) =

    if (bias != null) {

        if (input.numColumns() == 1) {

            weights.multAdd(input, bias.copy())
        }
        else {
            weights.multAdd(input, expandBias(bias, input.numColumns()))
        }

    }
    else {

        weights.mult(input, DenseMatrix(weights.numRows(), input.numColumns()))
    }

fun expandBias(bias: Matrix, inputColumns: Int): DenseMatrix {

    val biasRows = bias.numRows()

    val expandedBiasMatrix = createDenseMatrix(biasRows, inputColumns)

    for (indexRow in 0..biasRows - 1) {

        val biasColumns = bias.numColumns()

        for (indexColumn in 0..biasColumns - 1) {

            expandedBiasMatrix.set(indexRow, indexColumn, bias.get(indexRow, 0))

        }
    }

    return expandedBiasMatrix

}

fun oneHot(size : Int, oneHotIndex: Int, value : Double = 1.0) =

    DoubleArray(size) { index ->

        if (index == oneHotIndex) value else 0.0

    }
