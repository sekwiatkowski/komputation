package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.functions.backwardExpansionForConvolution
import shape.komputation.cpu.functions.convolutionsPerColumn
import shape.komputation.cpu.functions.convolutionsPerRow
import shape.komputation.cpu.functions.expandForConvolution
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuExpansionLayer internal constructor(name : String? = null, private val filterWidth: Int, private val filterHeight: Int) : BaseCpuForwardLayer(name) {

    private var numberInputRows = -1
    private var numberInputColumns = -1

    private var convolutionsPerRow = -1

    /*
        Ex.:
        input:
        i_11 i_12 i_13
        i_21 i_22 i_23
        i_31 i_32 i_33

        expansion:
        i_11 i_12
        i_21 i_22
        i_31 i_32
        i_12 i_13
        i_22 i_23
        i_32 i_33
    */
    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns

        this.convolutionsPerRow = convolutionsPerRow(this.numberInputColumns, this.filterWidth)
        val convolutionsPerColumn = convolutionsPerColumn(this.numberInputRows, this.filterHeight)

        val expanded = expandForConvolution(input.entries, this.numberInputRows, this.convolutionsPerRow, convolutionsPerColumn, filterWidth, filterHeight)

        return expanded

    }

    // d expansion / d input
    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        val summedDerivatives = backwardExpansionForConvolution(
            this.numberInputRows,
            this.numberInputColumns,
            this.filterHeight,
            this.convolutionsPerRow,
            chain.entries,
            chain.numberRows,
            chain.numberColumns)

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, summedDerivatives)

    }

}