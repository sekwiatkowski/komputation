package shape.konvolution.layers.continuation

import shape.konvolution.functions.sigmoid
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix

class SigmoidLayer(name : String? = null) : ContinuationLayer(name, 1, 0) {

    override fun forward() {

        this.lastForwardResult[0] = sigmoid(this.lastInput!!)

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : RealMatrix) {

        val lastForwardResult = this.lastForwardResult.single()

        val numberRows = lastForwardResult.numberRows()
        val nmberColumns = lastForwardResult.numberColumns()

        val gradient = createRealMatrix(numberRows, nmberColumns)

        for (indexRow in 0..numberRows - 1) {

            for (indexColumn in 0..nmberColumns - 1) {

                val forward = lastForwardResult.get(indexRow, indexColumn)

                val chainEntry = chain.get(indexRow, indexColumn)
                val dActivationWrtPreActivation = forward * (1 - forward)

                val derivative = chainEntry * dActivationWrtPreActivation

                gradient.set(indexRow, indexColumn, derivative)

            }
        }

        this.lastBackwardResultWrtInput = gradient

    }

}