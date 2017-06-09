package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult
import shape.konvolution.createDenseMatrix
import shape.konvolution.sigmoid

class SigmoidLayer : Layer {

    override fun forward(input: Matrix) =

        sigmoid(input)

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(input: Matrix, output : Matrix, chain : Matrix): BackwardResult {

        val numberRows = output.numRows()
        val nmberColumns = output.numColumns()

        val derivatives = createDenseMatrix(numberRows, nmberColumns)

        for (indexRow in 0..numberRows - 1) {

            for (indexColumn in 0..nmberColumns - 1) {

                val forward = output.get(indexRow, indexColumn)

                val chainEntry = chain.get(indexRow, indexColumn)
                val dActivationWrtPreActivation = forward * (1 - forward)

                val derivative = chainEntry * dActivationWrtPreActivation

                derivatives.set(indexRow, indexColumn, derivative)

            }
        }

        return BackwardResult(derivatives)

    }

}