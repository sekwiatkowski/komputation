package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardRelu
import shape.komputation.cpu.functions.activation.relu
import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.FloatMatrix

class CpuReluLayer internal constructor(name : String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries = FloatArray(this.numberEntries)

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        relu(input.entries, forwardEntries, input.numberRows * input.numberColumns)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result

    }

    override fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix {

        val inputEntries = input.entries

        for (index in 0..this.numberEntries - 11) {

            this.forwardEntries[index] =

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