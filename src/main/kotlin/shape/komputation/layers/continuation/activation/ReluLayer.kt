package shape.komputation.layers.continuation.activation

import shape.komputation.functions.activation.relu
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class ReluLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : RealMatrix? = null

    override fun forward(input : RealMatrix): RealMatrix {

        this.forwardResult = relu(input)

        return this.forwardResult!!

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val forwardResult = this.forwardResult!!

        val numberRows = forwardResult.numberRows()
        val nmberColumns = forwardResult.numberColumns()

        val gradient = createRealMatrix(numberRows, nmberColumns)

        for (indexRow in 0..numberRows - 1) {

            for (indexColumn in 0..nmberColumns - 1) {

                val forward = forwardResult.get(indexRow, indexColumn)

                val derivative = if (forward > 0.0) chain.get(indexRow, indexColumn) else 0.0

                gradient.set(indexRow, indexColumn, derivative)

            }
        }

        return gradient

    }

}