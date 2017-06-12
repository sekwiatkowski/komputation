package shape.konvolution.layers.entry

import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix

abstract class EntryPoint(private val name : String? = null) {

    var lastInput : Matrix? = null

    var lastForwardResult : RealMatrix? = null

    fun setInput(input: Matrix?) {

        this.lastInput = input
    }

    abstract fun forward()

}