package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.matrix.createRealMatrix
import shape.konvolution.matrix.sigmoid

class SigmoidLayer : ContinuationLayer {

    override fun forward(input: RealMatrix) =

        arrayOf(sigmoid(input))

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix): BackwardResult {

        val output = outputs.last()

        val numberRows = output.numberRows()
        val nmberColumns = output.numberColumns()

        val derivatives = createRealMatrix(numberRows, nmberColumns)

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