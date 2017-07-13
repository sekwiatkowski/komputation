package shape.komputation.cpu.forward.activation

import shape.komputation.cpu.forward.dropout.DropoutCompliant
import shape.komputation.functions.activation.differentiateSigmoid
import shape.komputation.functions.activation.sigmoid
import shape.komputation.functions.hadamard
import shape.komputation.matrix.DoubleMatrix

class CpuSigmoidLayer internal constructor(name : String? = null) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries : DoubleArray = DoubleArray(0)

    private var differentiation : DoubleArray? = null

    override fun forward(input : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        this.differentiation = null

        val result = DoubleMatrix(input.numberRows, input.numberColumns, sigmoid(input.entries))

        this.forwardEntries = result.entries

        return result

    }

    override fun forward(input: DoubleMatrix, mask: BooleanArray): DoubleMatrix {

        this.differentiation = null

        val inputEntries = input.entries

        this.forwardEntries = DoubleArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) sigmoid(inputEntries[index]) else 0.0

        }

        return DoubleMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

    }

    /*
        input = pre-activation
        output = activation

        d activation / d pre-activation = activation * (1 - activation)
     */
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        if (this.differentiation == null) {

            this.differentiation = differentiateSigmoid(this.forwardEntries)

        }

        return DoubleMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, differentiation!!))

    }

}