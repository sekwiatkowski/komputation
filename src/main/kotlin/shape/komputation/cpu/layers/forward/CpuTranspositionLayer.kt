package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.transpose
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuTranspositionLayer internal constructor(name : String? = null, numberEntries : Int) : BaseCpuForwardLayer(name)  {

    private val forwardEntries = FloatArray(numberEntries)
    private val backwardEntries = FloatArray(numberEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        transpose(input.numberRows, input.numberColumns, input.entries, this.forwardEntries)

        return FloatMatrix(input.numberColumns, input.numberRows, this.forwardEntries)

    }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        transpose(chain.numberRows, chain.numberColumns, chain.entries, this.backwardEntries)

        return FloatMatrix(chain.numberColumns, chain.numberRows, this.backwardEntries)

    }

}