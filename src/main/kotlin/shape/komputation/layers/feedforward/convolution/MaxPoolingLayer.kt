package shape.komputation.layers.feedforward.convolution

import shape.komputation.functions.convolution.maxPooling
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

/*
    Example:
    The input is a 20*100 matrix (filter size of 3, 22 words, 100 filters)
    The output is a 100D row vector
 */

class MaxPoolingLayer(name : String? = null) : FeedForwardLayer(name) {

    var input : RealMatrix? = null

    override fun forward(input : RealMatrix) : RealMatrix {

        this.input = input

        return maxPooling(input)

    }

    override fun backward(chain : RealMatrix): RealMatrix {

        val input = this.input!!

        val gradient = createRealMatrix(input.numberRows(), input.numberColumns())

        for (indexRow in 0..input.numberRows() - 1) {

            var maxValue = Double.NEGATIVE_INFINITY
            var maxIndexColumn = -1

            for (indexColumn in 0..input.numberColumns() - 1) {

                val entry = input.get(indexRow, indexColumn)

                if (entry > maxValue) {
                    maxValue = entry
                    maxIndexColumn = indexColumn

                }

            }

            gradient.set(indexRow, maxIndexColumn, chain.get(indexRow, 0))

        }

        return gradient

    }

}