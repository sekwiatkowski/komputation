package shape.komputation.layers.entry

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

abstract class EntryPoint(private val name : String? = null) {

    abstract fun forward(input: Matrix) : RealMatrix

}