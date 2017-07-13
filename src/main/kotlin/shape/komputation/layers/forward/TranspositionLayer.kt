package shape.komputation.layers.forward

import shape.komputation.functions.transpose
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.matrix.DoubleMatrix

class CpuTranspositionLayer internal constructor(name : String? = null) : BaseForwardLayer(name)  {

    override fun forward(input: DoubleMatrix, isTraining : Boolean) =

        DoubleMatrix(input.numberColumns, input.numberRows, transpose(input.numberRows, input.numberColumns, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberColumns, chain.numberRows, transpose(chain.numberRows, chain.numberColumns, chain.entries))

}

class TranspositionLayer(private val name : String? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuTranspositionLayer(this.name)


}

fun transpositionLayer(name : String? = null) = TranspositionLayer(name)