package shape.komputation.layers.continuation.activation

import shape.komputation.functions.activation.softmax
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class SoftmaxLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : RealMatrix? = null

    override fun forward(input : RealMatrix) : RealMatrix {

        this.forwardResult = softmax(input)

        return this.forwardResult!!

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : RealMatrix): RealMatrix {

        val lastForwardResult = this.forwardResult!!

        val gradient = createRealMatrix(lastForwardResult.numberRows(), lastForwardResult.numberColumns())

        for (indexColumn in 0..lastForwardResult.numberColumns() - 1) {

            for (outerIndexRow in 0..lastForwardResult.numberRows() - 1) {

                var derivative = 0.0

                val prediction = lastForwardResult.get(outerIndexRow, indexColumn)

                // Go through each row
                for (innerIndexRow in 0..lastForwardResult.numberRows() - 1) {

                    val chainEntry = chain.get(innerIndexRow, indexColumn)

                    // i == j
                    if (outerIndexRow == innerIndexRow) {

                        derivative += chainEntry * prediction * (1 - prediction)

                    }
                    // i != j
                    else {

                        val otherPrediction = lastForwardResult.get(innerIndexRow, indexColumn)

                        derivative += chainEntry * (-prediction * otherPrediction)

                    }

                }

                gradient.set(outerIndexRow, indexColumn, derivative)

            }

        }

        return gradient

    }
}