package shape.komputation.cpu.layers.forward.activation

import shape.komputation.cpu.functions.activation.differentiateTanh
import shape.komputation.cpu.functions.activation.tanh
import shape.komputation.cpu.functions.hadamard
import shape.komputation.matrix.FloatMatrix

class CpuTanhLayer internal constructor(name: String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuActivationLayer(name) {

    private val numberEntries = this.numberRows * this.numberColumns

    private val forwardEntries = FloatArray(this.numberEntries)
    private val backwardEntries = FloatArray(this.numberEntries)

    private var differentiation = FloatArray(this.numberEntries)
    private var hasCachedDifferentiation = false

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.hasCachedDifferentiation = false

        tanh(input.entries, this.forwardEntries, this.numberEntries)

        val result = FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

        return result
    }

    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        if (!this.hasCachedDifferentiation) {

            differentiateTanh(this.forwardEntries, this.differentiation, this.numberEntries)

            this.hasCachedDifferentiation = true

        }

        hadamard(chain.entries, this.differentiation, this.backwardEntries, this.numberEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.backwardEntries)

    }

}