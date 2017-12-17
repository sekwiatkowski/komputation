package com.komputation.cpu.layers.recurrent.extraction

import com.komputation.cpu.functions.add
import com.komputation.cpu.functions.getColumn
import com.komputation.cpu.functions.setColumn
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.computeNumberPossibleLengths
import com.komputation.cpu.layers.computePossibleLengths
import com.komputation.cpu.layers.recurrent.series.Series

class AllSteps(
    private val series: Series,
    private val hiddenDimension: Int,
    private val minimumSteps : Int,
    private val maximumSteps : Int) : ResultExtractionStrategy {

    override var forwardResult = FloatArray(0)
    private val numberOutputRows = this.hiddenDimension
    override var numberOutputColumns = -1

    private val numberPossibleOutputLengths = computeNumberPossibleLengths(this.minimumSteps, this.maximumSteps)
    override val possibleOutputLengths = computePossibleLengths(this.minimumSteps, numberPossibleOutputLengths)

    private val store = VariableLengthFloatArray(this.hiddenDimension, this.possibleOutputLengths)

    override fun extractResult(numberInputColumns : Int): FloatArray {
        this.numberOutputColumns = numberInputColumns
        this.forwardResult = this.store.get(numberInputColumns)

        for (index in 0 until numberInputColumns) {
            setColumn(this.series.getForwardResult(index), index, this.numberOutputRows, this.forwardResult)
        }

        return this.forwardResult
    }

    private val stepChain = FloatArray(this.hiddenDimension)

    override fun backwardStep(step : Int, chain: FloatArray, previousBackwardPreviousHiddenState : FloatArray?): FloatArray {
        getColumn(chain, step, this.hiddenDimension, this.stepChain)

        if(previousBackwardPreviousHiddenState != null) {
            add(this.stepChain, previousBackwardPreviousHiddenState, this.stepChain, this.hiddenDimension)
        }

        return stepChain
    }

}