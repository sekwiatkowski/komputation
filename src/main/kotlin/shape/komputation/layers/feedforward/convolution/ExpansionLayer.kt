package shape.komputation.layers.feedforward.convolution

import shape.komputation.functions.backwardExpansion
import shape.komputation.functions.expand
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix

class ExpansionLayer(name : String? = null, private val filterWidth: Int, private val filterHeight: Int) : ContinuationLayer(name) {

    private var numberInputRows = -1
    private var numberInputColumns = -1

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        return expand(input.entries, this.numberInputRows, this.numberInputColumns, filterWidth, filterHeight)

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val summedDerivatives = backwardExpansion(this.numberInputRows, this.numberInputColumns, filterWidth, chain.entries, chain.numberRows, chain.numberColumns)

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, summedDerivatives)

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