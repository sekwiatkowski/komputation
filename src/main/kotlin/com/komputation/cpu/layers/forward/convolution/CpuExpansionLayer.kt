package com.komputation.cpu.layers.forward.convolution

import com.komputation.cpu.functions.backwardExpansionForConvolution
import com.komputation.cpu.functions.computeNumberFilterColumnPositions
import com.komputation.cpu.functions.expandForConvolution
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer

class CpuExpansionLayer internal constructor(
    name: String? = null,
    numberInputRows: Int,
    minimumColumns: Int,
    maximumColumns: Int,
    private val numberFilterRowPositions: Int,
    filterLength: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : BaseCpuVariableLengthForwardLayer(
    name,
    numberInputRows,
    filterLength,
    minimumColumns,
    maximumColumns,
    { inputLength : Int -> computeNumberFilterColumnPositions(inputLength, filterWidth) * numberFilterRowPositions }) {

    private var numberFilterColumnPositions = -1

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
    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        this.numberFilterColumnPositions = computeNumberFilterColumnPositions(numberInputColumns, this.filterWidth)

        expandForConvolution(
            this.numberInputRows,
            input,
            this.filterWidth,
            this.filterHeight,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositions,
            forwardResult)
    }

    // d expansion / d input
    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardExpansionForConvolution(
            this.numberInputRows,
            backwardResult,
            this.filterHeight,
            this.numberFilterRowPositions,
            this.numberFilterColumnPositions,
            chain,
            this.numberOutputRows)
    }

}