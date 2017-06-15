package shape.komputation.layers.feedforward.activation

import shape.komputation.functions.activation.relu
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix

class ReluLayer(name : String? = null) : ActivationLayer(name) {

    private var forwardResult : RealMatrix? = null

    override fun forward(input : RealMatrix): RealMatrix {

        this.forwardResult = createRealMatrix(input.numberRows(), input.numberColumns(), relu(input.getEntries()))

        return this.forwardResult!!

    }

    override fun backward(chain : RealMatrix) : RealMatrix {

        val forwardEntries = this.forwardResult!!.getEntries()
        val chainEntries = chain.getEntries()

        val backwardEntries = DoubleArray(chainEntries.size) { index ->

            val forwardEntry = forwardEntries[index]

            if (forwardEntry > 0.0)
                chainEntries[index]
            else
                0.0

        }

        return createRealMatrix(chain.numberRows(), chain.numberColumns(), backwardEntries)

    }

}