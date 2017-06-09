package shape.konvolution.layers

import no.uib.cipr.matrix.Matrix
import shape.konvolution.BackwardResult

interface Layer {

    fun forward(input : Matrix) : Matrix

    fun backward(input: Matrix, output : Matrix, chain : Matrix) : BackwardResult

}