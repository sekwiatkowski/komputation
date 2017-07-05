package shape.komputation.layers.forward.activation

import shape.komputation.functions.activation.backwardRelu
import shape.komputation.functions.activation.relu
import shape.komputation.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.DoubleMatrix

class ReluLayer internal constructor(name : String? = null) : ActivationLayer(name), DropoutCompliant {

    private var forwardEntries : DoubleArray = DoubleArray(0)

    override fun forward(input : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val result = DoubleMatrix(input.numberRows, input.numberColumns, relu(input.entries))

        this.forwardEntries = result.entries

        return result

    }

    override fun forward(input: DoubleMatrix, mask: BooleanArray): DoubleMatrix {

        val inputEntries = input.entries

        val forwardEntries = DoubleArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) {
                relu(inputEntries[index])
            }
            else {
                0.0
            }

        }

        val result = DoubleMatrix(input.numberRows, input.numberColumns, forwardEntries)

        this.forwardEntries = forwardEntries

        return result

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainEntries = chain.entries

        val backwardEntries = backwardRelu(this.forwardEntries, chainEntries)

        return DoubleMatrix(chain.numberRows, chain.numberColumns, backwardEntries)

    }

}

fun reluLayer(name : String? = null) = ReluLayer(name)