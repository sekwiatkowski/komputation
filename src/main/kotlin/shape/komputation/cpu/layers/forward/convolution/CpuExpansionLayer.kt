package shape.komputation.cpu.layers.forward.convolution

import shape.komputation.cpu.functions.backwardExpansionForConvolution
import shape.komputation.cpu.functions.computeNumberFilterColumnPositions
import shape.komputation.cpu.functions.expandForConvolution
import shape.komputation.cpu.functions.findFirstPaddedColumn
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.matrix.FloatMatrix

class CpuExpansionLayer internal constructor(
    name : String? = null,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val numberConvolutions : Int,
    private val numberFilterRowPositions: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : BaseCpuForwardLayer(name) {

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
    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    private val filterLength = this.filterWidth * this.filterHeight

    private val numberForwardEntries = this.filterLength * this.numberConvolutions
    private val forwardEntries = FloatArray(this.numberForwardEntries)

    private val backwardEntries = FloatArray(this.numberInputEntries)

    private var numberActualColumns = -1
    private var numberFilterColumnPositions = -1

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        val inputEntries = input.entries

        val firstPaddedColumn = findFirstPaddedColumn(inputEntries, this.numberInputRows, this.numberInputColumns)

        this.numberActualColumns = if(firstPaddedColumn == -1) this.numberInputColumns else firstPaddedColumn

        this.numberFilterColumnPositions = computeNumberFilterColumnPositions(this.numberActualColumns, this.filterWidth)

        expandForConvolution(
            input.entries,
            this.numberInputRows,
            this.forwardEntries,
            this.numberForwardEntries,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositions,
            this.filterWidth,
            this.filterHeight)

        return FloatMatrix(this.filterLength, this.numberConvolutions, this.forwardEntries)

    }

    // d expansion / d input
    override fun backward(withinBatch : Int, chain : FloatMatrix): FloatMatrix {

        val numberActualEntries = this.numberActualColumns * this.numberInputRows

        backwardExpansionForConvolution(
            this.numberInputRows,
            numberActualEntries,
            this.numberInputEntries,
            this.backwardEntries,
            this.filterHeight,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositions,
            chain.entries,
            chain.numberRows)

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, this.backwardEntries)

    }

}