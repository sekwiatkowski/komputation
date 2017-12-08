package com.komputation.cpu.loss

abstract class BaseCpuLossFunction(
    override val numberInputRows: Int,
    private val maximumLength: Int,
    private val hasFixedLength : Boolean) : CpuLossFunction {

    override var backwardResult = FloatArray(0)
    override var numberInputColumns = -1

    private val minimumLength = if(this.hasFixedLength) maximumLength else 1
    private val numberPossibleLengths = this.maximumLength - this.minimumLength + 1
    private val possibleLengths = Array(this.maximumLength) { index -> index + this.minimumLength }

    private val backwardResultsOverPossibleLengths = Array(this.numberPossibleLengths) { index -> FloatArray(this.possibleLengths[index] * this.numberInputRows) }

    override fun forward(numberInputColumns : Int, predictions: FloatArray, targets : FloatArray): Float {
        this.numberInputColumns = numberInputColumns

        return computeLoss(targets, predictions)
    }

    abstract fun computeLoss(targets: FloatArray, predictions: FloatArray): Float

    override fun backward(predictions: FloatArray, targets : FloatArray): FloatArray {
        this.backwardResult = this.backwardResultsOverPossibleLengths[this.numberInputColumns - this.minimumLength]

        computeDifferentation(targets, predictions, backwardResult)

        return this.backwardResult
    }

    abstract protected fun computeDifferentation(targets: FloatArray, predictions: FloatArray, result : FloatArray)

}