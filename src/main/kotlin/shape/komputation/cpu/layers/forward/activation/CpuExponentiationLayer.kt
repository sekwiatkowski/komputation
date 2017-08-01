package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.exponentiate
import shape.komputation.cpu.functions.hadamard
import shape.komputation.matrix.FloatMatrix

class CpuExponentiationLayer internal constructor(name : String? = null, private val numberEntries : Int) : BaseCpuActivationLayer(name) {

    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        exponentiate(input.entries, this.forwardEntries, input.numberRows * input.numberColumns)

        val result = FloatMatrix(input.numberRows, input.numberColumns, this.forwardEntries)

        return result

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        hadamard(chain.entries, this.forwardEntries, this.backwardEntries, this.numberEntries)

        val result = FloatMatrix(chain.numberRows, chain.numberColumns, this.backwardEntries)

        return result

    }

}