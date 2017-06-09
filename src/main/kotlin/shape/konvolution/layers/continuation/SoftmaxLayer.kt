package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.RealMatrix
import shape.konvolution.createRealMatrix
import shape.konvolution.softmax

class SoftmaxLayer : ContinuationLayer {

    override fun forward(input: RealMatrix) =

        softmax(input)

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) =

        createRealMatrix(output.numberRows(), output.numberColumns())
            .let { derivatives ->

                for (indexColumn in 0..output.numberColumns() - 1) {

                    for (outerIndexRow in 0..output.numberRows() - 1) {

                        var derivative = 0.0

                        val prediction = output.get(outerIndexRow, indexColumn)

                        // Go through each row
                        for (innerIndexRow in 0..output.numberRows() - 1) {

                            val chainEntry = chain.get(innerIndexRow, indexColumn)

                            // i == j
                            if (outerIndexRow == innerIndexRow) {

                                derivative += chainEntry * prediction * (1 - prediction)

                            }
                            // i != j
                            else {

                                val otherPrediction = output.get(innerIndexRow, indexColumn)

                                derivative += chainEntry * (-prediction * otherPrediction)

                            }

                        }

                        derivatives.set(outerIndexRow, indexColumn, derivative)

                    }

                }

                derivatives

            }
            .let { derivatives ->

                BackwardResult(derivatives)

            }
}