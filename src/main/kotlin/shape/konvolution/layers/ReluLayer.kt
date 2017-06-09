package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult
import shape.konvolution.createDenseMatrix
import shape.konvolution.relu

class ReluLayer : Layer {

    override fun forward(input: Matrix) =

        relu(input)

    override fun backward(input: Matrix, output : Matrix, chain : Matrix): BackwardResult {

        val numberRows = output.numRows()
        val nmberColumns = output.numColumns()

        val derivatives = createDenseMatrix(numberRows, nmberColumns)

        for (indexRow in 0..numberRows - 1) {

            for (indexColumn in 0..nmberColumns - 1) {

                val forward = output.get(indexRow, indexColumn)

                val derivative = if (forward > 0.0) chain.get(indexRow, indexColumn) else 0.0

                derivatives.set(indexRow, indexColumn, derivative)

            }
        }

        return BackwardResult(derivatives)


    }

}