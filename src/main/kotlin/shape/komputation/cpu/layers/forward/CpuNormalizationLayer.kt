package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.activation.backwardNormalization
import shape.komputation.cpu.functions.activation.normalize
import shape.komputation.cpu.layers.forward.activation.BaseCpuActivationLayer
import shape.komputation.matrix.FloatMatrix

/*
    a/(a+b+c)

    input entry = a
    forward entry = a/(a+b+c)
 */
class CpuNormalizationLayer internal constructor(name : String? = null, private val numberRows : Int, private val numberColumns : Int) : BaseCpuActivationLayer(name) {

    private var inputEntries = FloatArray(this.numberRows * this.numberColumns)
    private val forwardEntries = FloatArray(this.numberRows * this.numberColumns)
    private val sumEntries = FloatArray(this.numberColumns)
    private val backwardEntries = FloatArray(this.numberRows * this.numberColumns)

    override fun forward(input : FloatMatrix, isTraining : Boolean): FloatMatrix {

        this.inputEntries = input.entries

        normalize(this.numberRows, this.numberColumns, inputEntries, this.sumEntries, this.forwardEntries)

        val result = FloatMatrix(this.numberRows, this.numberColumns, this.forwardEntries)

        return result

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        backwardNormalization(this.numberRows, this.numberColumns, chain.entries, this.inputEntries, this.forwardEntries, this.sumEntries, this.backwardEntries)

        return FloatMatrix(this.numberRows, this.numberColumns, this.backwardEntries)

    }

}