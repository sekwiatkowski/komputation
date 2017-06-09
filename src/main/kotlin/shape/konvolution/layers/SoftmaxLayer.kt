package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult
import shape.konvolution.createDenseMatrix
import shape.konvolution.softmax

class SoftmaxLayer : Layer {

    override fun forward(input: Matrix) =

        softmax(input)

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(input: Matrix, output : Matrix, chain : Matrix) =

        createDenseMatrix(output.numRows(), output.numColumns())
            .let { derivatives ->

                for (indexColumn in 0..output.numColumns() - 1) {

                    for (outerIndexRow in 0..output.numRows() - 1) {

                        var derivative = 0.0

                        val prediction = output.get(outerIndexRow, indexColumn)

                        // Go through each row
                        for (innerIndexRow in 0..output.numRows() - 1) {

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