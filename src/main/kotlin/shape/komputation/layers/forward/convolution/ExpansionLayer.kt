package shape.komputation.layers.forward.convolution

import shape.komputation.functions.backwardExpansionForConvolution
import shape.komputation.functions.expandForConvolution
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix

class ExpansionLayer internal constructor(name : String? = null, private val filterWidth: Int, private val filterHeight: Int) : ForwardLayer(name) {

    private var numberInputRows = -1
    private var numberInputColumns = -1

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        return expandForConvolution(input.entries, this.numberInputRows, this.numberInputColumns, filterWidth, filterHeight)

    }

    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        val summedDerivatives = backwardExpansionForConvolution(this.numberInputRows, this.numberInputColumns, filterWidth, chain.entries, chain.numberRows, chain.numberColumns)

        return DoubleMatrix(this.numberInputRows, this.numberInputColumns, summedDerivatives)

    }

}


fun expansionLayer(
    filterWidth: Int,
    filterHeight: Int): ExpansionLayer {

    return expansionLayer(null, filterWidth, filterHeight)
}

fun expansionLayer(
    name : String?,
    filterWidth: Int,
    filterHeight: Int): ExpansionLayer {

    return ExpansionLayer(name, filterWidth, filterHeight)
}