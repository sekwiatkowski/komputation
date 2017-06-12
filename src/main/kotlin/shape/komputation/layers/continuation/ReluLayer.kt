package shape.komputation.layers.continuation

import shape.komputation.functions.relu
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class ReluLayer(name : String? = null) : ContinuationLayer(name, 1, 0) {

    override fun forward() {

        this.lastForwardResult[0] = relu(this.lastInput!!)

    }

    override fun backward(chain : RealMatrix) {

        val lastForwardResult = this.lastForwardResult.single()

        val numberRows = lastForwardResult.numberRows()
        val nmberColumns = lastForwardResult.numberColumns()

        val gradient = createRealMatrix(numberRows, nmberColumns)

        for (indexRow in 0..numberRows - 1) {

            for (indexColumn in 0..nmberColumns - 1) {

                val forward = lastForwardResult.get(indexRow, indexColumn)

                val derivative = if (forward > 0.0) chain.get(indexRow, indexColumn) else 0.0

                gradient.set(indexRow, indexColumn, derivative)

            }
        }

        this.lastBackwardResultWrtInput = gradient

    }

}