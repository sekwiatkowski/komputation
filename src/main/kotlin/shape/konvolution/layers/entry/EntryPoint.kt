package shape.konvolution.layers.entry

import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix

interface EntryPoint {

    fun forward(input : Matrix) : RealMatrix

}