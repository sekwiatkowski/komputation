package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.functions.transpose
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.DoubleMatrix

class CpuTranspositionLayer internal constructor(name : String? = null) : BaseCpuForwardLayer(name)  {

    override fun forward(input: DoubleMatrix, isTraining : Boolean) =

        DoubleMatrix(input.numberColumns, input.numberRows, transpose(input.numberRows, input.numberColumns, input.entries))

    override fun backward(chain: DoubleMatrix) =

        DoubleMatrix(chain.numberColumns, chain.numberRows, transpose(chain.numberRows, chain.numberColumns, chain.entries))

}