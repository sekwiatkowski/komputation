package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.backwardRelu
import shape.komputation.cpu.functions.activation.relu
import shape.komputation.matrix.FloatMatrix

class CpuReluLayer internal constructor(name : String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuActivationLayer(name) {

    private val numberEntries = this.numberRows * this.numberColumns

    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        relu(input.entries, this.forwardEntries, this.numberEntries)

        val result = FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

        return result

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        backwardRelu(this.forwardEntries, chain.entries, this.backwardEntries, this.numberEntries)

        return FloatMatrix(numberRows, numberColumns, this.backwardEntries)

    }

}