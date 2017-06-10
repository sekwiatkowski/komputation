package shape.konvolution.layers.continuation

import shape.konvolution.matrix.RealMatrix

abstract class ContinuationLayer(resultSize : Int, parameterSize : Int) {

    var lastInput : RealMatrix? = null

    var lastForwardResult : Array<RealMatrix?> = arrayOfNulls(resultSize)

    var lastBackwardResultWrtInput : RealMatrix? = null
    var lastBackwardResultWrtParameters : Array<RealMatrix?> = arrayOfNulls(parameterSize)

    fun setInput(input: RealMatrix) {

        this.lastInput = input

    }

    abstract fun forward()

    abstract fun backward(chain : RealMatrix)

}