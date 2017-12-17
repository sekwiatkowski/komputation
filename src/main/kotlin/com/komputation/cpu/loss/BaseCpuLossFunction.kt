package com.komputation.cpu.loss

import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.computeNumberPossibleLengths
import com.komputation.cpu.layers.computePossibleLengths

abstract class BaseCpuLossFunction(
    val numberInputRows: Int,
    private val minimumLength : Int,
    private val maximumLength: Int) : CpuLossFunction {

    override var backwardResult = FloatArray(0)
    override var numberInputColumns = -1

    private val numberPossibleInputLengths = computeNumberPossibleLengths(this.minimumLength, this.maximumLength)
    private val possibleInputLengths = computePossibleLengths(this.minimumLength, this.numberPossibleInputLengths)

    private val backwardResultsOverPossibleLengths = VariableLengthFloatArray(this.numberInputRows, this.possibleInputLengths)

    override fun forward(numberInputColumns : Int, predictions: FloatArray, targets : FloatArray): Float {
        this.numberInputColumns = numberInputColumns

        return computeLoss(targets, predictions)
    }

    abstract fun computeLoss(targets: FloatArray, predictions: FloatArray): Float

    override fun backward(predictions: FloatArray, targets : FloatArray): FloatArray {
        this.backwardResult = this.backwardResultsOverPossibleLengths.get(this.numberInputColumns)

        computeDifferentation(targets, predictions, backwardResult)

        return this.backwardResult
    }

    abstract protected fun computeDifferentation(targets: FloatArray, predictions: FloatArray, result : FloatArray)

}