package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardRelu
import shape.komputation.cpu.functions.activation.relu
import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.FloatMatrix

class CpuReluLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries = FloatArray(0)

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        val result = FloatMatrix(input.numberRows, input.numberColumns, relu(input.entries))

        this.forwardEntries = result.entries

        return result

    }

    override fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix {

        val inputEntries = input.entries

        val forwardEntries = FloatArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) {
                relu(inputEntries[index])
            }
            else {
                0.0f
            }

        }

        val result = FloatMatrix(input.numberRows, input.numberColumns, forwardEntries)

        this.forwardEntries = forwardEntries

        return result

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        val chainEntries = chain.entries

        val backwardEntries = backwardRelu(this.forwardEntries, chainEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, backwardEntries)

    }

}