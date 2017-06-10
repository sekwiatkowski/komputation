package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

/*
    Example:
    The input is a 20*100 matrix (filter size of 3, 22 words, 100 filters)
    The output is a 100D row vector
 */

class MaxPoolingLayer : ContinuationLayer {

    override fun forward(input: RealMatrix): RealMatrix {

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

        return maxPooled

    }

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix): BackwardResult {

        val derivatives = createRealMatrix(input.numberRows(), input.numberColumns())

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
            derivatives.set(indexRow, maxIndexColumn, chain.get(indexRow, 0))

        }

        return BackwardResult(derivatives)

    }

}