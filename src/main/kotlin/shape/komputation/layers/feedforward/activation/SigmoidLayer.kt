package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.sigmoid
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class SigmoidLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : RealMatrix? = null

    override fun forward(input : RealMatrix): RealMatrix {

        this.forwardResult = sigmoid(input)

        return this.forwardResult!!

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : RealMatrix) : RealMatrix {

        val lastForwardResult = this.forwardResult!!

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

        return gradient

    }

}