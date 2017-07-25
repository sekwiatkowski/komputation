package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardRelu
import shape.komputation.cpu.functions.activation.relu
import shape.komputation.matrix.FloatMatrix

class CpuReluLayer internal constructor(name : String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name) {

    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        relu(input.entries, this.forwardEntries, input.numberRows * input.numberColumns)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        backwardRelu(this.forwardEntries, chain.entries, this.backwardEntries, this.numberEntries)

        return FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

    }

}