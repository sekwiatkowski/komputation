package shape.konvolution.layers

import no.uib.cipr.matrix.DenseMatrix
import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult
import shape.konvolution.createDenseMatrix

/*
    Example:
    The input is a 20*100 matrix (filter size of 3, 22 words, 100 filters)
    The output is a 100D row vector
 */

class MaxPoolingLayer : Layer {

    override fun forward(input: Matrix): DenseMatrix {

        val maxPooled = createDenseMatrix(input.numRows(), 1)

        var index = 0

        for (indexRow in 0..input.numRows() - 1) {

            var maxValue = Double.NEGATIVE_INFINITY

            for (indexColumn in 0..input.numColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                if (entry > maxValue) {
                    maxValue = entry
                }

            }

            maxPooled.set(index++, 0, maxValue)

        }

        return maxPooled

    }

    override fun backward(input: Matrix, output : Matrix, chain : Matrix): BackwardResult {

        val derivatives = createDenseMatrix(input.numRows(), input.numColumns())

        for (indexRow in 0..input.numRows() - 1) {

            var maxValue = Double.NEGATIVE_INFINITY
            var maxIndexColumn = -1

            for (indexColumn in 0..input.numColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                if (entry > maxValue) {
                    maxValue = entry
                    maxIndexColumn = indexColumn
                }

            }
            derivatives.set(indexRow, maxIndexColumn, chain.get(indexRow, 0))

        }

        return BackwardResult(derivatives)

    }

}