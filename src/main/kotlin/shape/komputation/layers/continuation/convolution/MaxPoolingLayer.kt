package shape.komputation.layers.continuation.convolution

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

/*
    Example:
    The input is a 20*100 matrix (filter size of 3, 22 words, 100 filters)
    The output is a 100D row vector
 */

class MaxPoolingLayer(name : String? = null) : ContinuationLayer(name, 1, 0) {

    override fun forward() {

        val input = this.lastInput!!

        val maxPooled = createRealMatrix(input.numberRows(), 1)

        var index = 0

        for (indexRow in 0..input.numberRows() - 1) {

            var maxValue = Double.NEGATIVE_INFINITY

            for (indexColumn in 0..input.numberColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                if (entry > maxValue) {
                    maxValue = entry
                }

            }

            maxPooled.set(index++, 0, maxValue)

        }

        this.lastForwardResult[0] = maxPooled

    }

    override fun backward(chain : RealMatrix) {

        val lastInput = this.lastInput!!

        val derivatives = createRealMatrix(lastInput.numberRows(), lastInput.numberColumns())

        for (indexRow in 0..lastInput.numberRows() - 1) {

            var maxValue = Double.NEGATIVE_INFINITY
            var maxIndexColumn = -1

            for (indexColumn in 0..lastInput.numberColumns() - 1) {

                val entry = lastInput.get(indexRow, indexColumn)

                if (entry > maxValue) {
                    maxValue = entry
                    maxIndexColumn = indexColumn

                }

            }

            derivatives.set(indexRow, maxIndexColumn, chain.get(indexRow, 0))

        }

        this.lastBackwardResultWrtInput = derivatives

    }

}