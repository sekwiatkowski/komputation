package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateSigmoid
import shape.komputation.cpu.functions.activation.sigmoid
import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.FloatMatrix

class CpuSigmoidLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries : FloatArray = FloatArray(0)

    private var differentiation : FloatArray? = null

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        this.differentiation = null

        val result = FloatMatrix(input.numberRows, input.numberColumns, sigmoid(input.entries))

        this.forwardEntries = result.entries

        return result

    }

    override fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix {

        this.differentiation = null

        val inputEntries = input.entries

        this.forwardEntries = FloatArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) sigmoid(inputEntries[index]) else 0.0f

        }

        return FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : FloatMatrix) : FloatMatrix {

        if (this.differentiation == null) {

            this.differentiation = differentiateSigmoid(this.forwardEntries)

        }

        return FloatMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, differentiation!!))

    }

}