package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.relu

class ReluLayer : ContinuationLayer {

    override fun forward(input: RealMatrix) =

        arrayOf(relu(input))

    override fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix): BackwardResult {

        val output = outputs.last()

        val numberRows = output.numberRows()
        val nmberColumns = output.numberColumns()

        val derivatives = createRealMatrix(numberRows, nmberColumns)

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