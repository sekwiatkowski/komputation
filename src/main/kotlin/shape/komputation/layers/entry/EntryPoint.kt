package shape.komputation.layers.entry

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

abstract class EntryPoint(private val name : String? = null) {

    var lastInput : Matrix? = null

    var lastForwardResult : RealMatrix? = null

    fun setInput(input: Matrix?) {

        this.lastInput = input
    }

    abstract fun forward()

}