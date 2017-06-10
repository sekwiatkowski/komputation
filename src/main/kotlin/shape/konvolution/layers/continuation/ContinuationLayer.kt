package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix

interface ContinuationLayer {

    fun forward(input : RealMatrix) : RealMatrix

    fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) : BackwardResult

}