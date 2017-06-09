package shape.konvolution.layers.entry

import shape.konvolution.BackwardResult
import shape.konvolution.Matrix
import shape.konvolution.RealMatrix

class InputLayer : EntryPoint {

    override fun forward(input : Matrix) : RealMatrix {

        return input as RealMatrix
    }

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix) : BackwardResult? {

        return null
    }

}