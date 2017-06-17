package shape.komputation.layers.feedforward.convolution

import shape.komputation.functions.backwardExpansion
import shape.komputation.functions.expand
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.matrix.DoubleMatrix

class ExpansionLayer(name : String? = null, private val filterWidth: Int, private val filterHeight: Int) : FeedForwardLayer(name) {

    private var input : DoubleMatrix? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.input = input

        return expand(input.entries, input.numberRows, input.numberColumns, filterWidth, filterHeight)

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val input = this.input!!

        val summedDerivatives = backwardExpansion(input.numberRows, input.numberColumns, filterWidth, chain.entries, chain.numberRows, chain.numberColumns)

        return DoubleMatrix(input.numberRows, input.numberColumns, summedDerivatives)

    }

}


fun createExpansionLayer(
    filterWidth: Int,
    filterHeight: Int): ExpansionLayer {

    return createExpansionLayer(null, filterWidth, filterHeight)
}

fun createExpansionLayer(
    name : String?,
    filterWidth: Int,
    filterHeight: Int): ExpansionLayer {

    return ExpansionLayer(name, filterWidth, filterHeight)
}