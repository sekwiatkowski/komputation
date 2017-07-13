package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.functions.activation.differentiateTanh
import shape.komputation.functions.activation.tanh
import shape.komputation.functions.hadamard
import shape.komputation.matrix.DoubleMatrix

class CpuTanhLayer internal constructor(name: String? = null) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries : DoubleArray = DoubleArray(0)

    private var differentiation : DoubleArray? = null

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        this.differentiation = null

        val result = DoubleMatrix(input.numberRows, input.numberColumns, tanh(input.entries))

        this.forwardEntries = result.entries

        return result
    }

    override fun forward(input: DoubleMatrix, mask: BooleanArray): DoubleMatrix {

        this.differentiation = null

        val inputEntries = input.entries

        this.forwardEntries = DoubleArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) tanh(inputEntries[index]) else 0.0

        }

        return DoubleMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        if (this.differentiation == null) {

            this.differentiation = differentiateTanh(this.forwardEntries)

        }

        return DoubleMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, differentiation!!))

    }

}