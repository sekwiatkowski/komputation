package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateTanh
import shape.komputation.cpu.functions.activation.tanh
import shape.komputation.cpu.functions.hadamard
import shape.komputation.cpu.layers.forward.dropout.DropoutCompliant
import shape.komputation.matrix.FloatMath
import shape.komputation.matrix.FloatMatrix

class CpuTanhLayer internal constructor(name: String? = null) : BaseCpuActivationLayer(name), DropoutCompliant {

    private var forwardEntries : FloatArray = FloatArray(0)

    private var differentiation : FloatArray? = null

    override fun forward(input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.differentiation = null

        val result = FloatMatrix(input.numberRows, input.numberColumns, tanh(input.entries))

        this.forwardEntries = result.entries

        return result
    }

    override fun forward(input: FloatMatrix, mask: BooleanArray): FloatMatrix {

        this.differentiation = null

        val inputEntries = input.entries

        this.forwardEntries = FloatArray(input.numberRows * input.numberColumns) { index ->

            if(mask[index]) FloatMath.tanh(inputEntries[index]) else 0.0f

        }

        return FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

    }

    /*
        Note that each pre-activation effects all nodes.
        For i == j: prediction (1 - prediction)
        for i != j: -(prediction_i * prediction_j)
     */
    override fun backward(chain : FloatMatrix): FloatMatrix {

        if (this.differentiation == null) {

            this.differentiation = differentiateTanh(this.forwardEntries)

        }

        return FloatMatrix(chain.numberRows, chain.numberColumns, hadamard(chain.entries, differentiation!!))

    }

}