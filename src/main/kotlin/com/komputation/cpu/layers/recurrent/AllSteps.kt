package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.functions.setColumn
import com.komputation.cpu.layers.CpuForwardState
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.computeNumberPossibleLengths
import com.komputation.cpu.layers.computePossibleLengths

class AllSteps(
    private val series: Series,
    numberRows : Int,
    private val minimumSteps : Int,
    private val maximumSteps : Int) : CpuForwardState {

    override var forwardResult = FloatArray(0)
    override val numberOutputRows = numberRows
    override var numberOutputColumns = -1

    private val numberPossibleLengths = computeNumberPossibleLengths(this.minimumSteps, this.maximumSteps)
    private val possibleLengths = computePossibleLengths(this.minimumSteps, numberPossibleLengths)

    private val store = VariableLengthFloatArray(numberRows, this.possibleLengths)

    fun extractResult(numberInputColumns : Int): FloatArray {
        this.numberOutputColumns = numberInputColumns
        this.forwardResult = this.store.get(numberInputColumns)

        for (index in 0 until numberInputColumns) {
            setColumn(this.series.getForwardResult(index), index, this.numberOutputRows, this.forwardResult)
        }

        return this.forwardResult
    }

}