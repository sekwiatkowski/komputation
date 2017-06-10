package shape.konvolution.layers.continuation

import shape.konvolution.BackwardResult
import shape.konvolution.matrix.RealMatrix

interface ContinuationLayer {

    fun forward(input : RealMatrix) : Array<RealMatrix>

    fun backward(inputs: Array<RealMatrix>, outputs : Array<RealMatrix>, chain : RealMatrix) : BackwardResult

}