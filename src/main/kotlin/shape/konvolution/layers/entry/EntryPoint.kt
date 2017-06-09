package shape.konvolution.layers.entry

import shape.konvolution.BackwardResult
import shape.konvolution.Matrix
import shape.konvolution.RealMatrix

interface EntryPoint {

    fun forward(input : Matrix) : RealMatrix

    fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) : BackwardResult?

}