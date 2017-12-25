package com.komputation.cpu.layers

fun computeNumberPossibleLengths(minimumLength: Int, maximumLength: Int) =
    maximumLength - minimumLength + 1

fun computeLengthIndex(length : Int, minimumLength: Int) =
    length - minimumLength

fun computePossibleLengths(minimumLength: Int, numberPossibleLengths : Int) =
    IntArray(numberPossibleLengths) { index -> index + minimumLength }

abstract class BaseCpuContinuationLayer(
    override val name: String?,
    final override val numberInputRows : Int,
    final override val numberOutputRows : Int,
    protected val minimumColumns : Int,
    protected val maximumColumns : Int,
    private val computeNumberOutputColumns : (Int) -> Int = { inputLength -> inputLength }) : CpuContinuation {

    override var backwardResult = FloatArray(0)
    override var numberInputColumns = -1

    override var forwardResult = FloatArray(0)
    override var numberOutputColumns = -1

    protected val numberPossibleInputLengths = computeNumberPossibleLengths(this.minimumColumns, this.maximumColumns)
    final override val possibleInputLengths = computePossibleLengths(this.minimumColumns, this.numberPossibleInputLengths)
    final override val possibleOutputLengths = this.possibleInputLengths.map(this.computeNumberOutputColumns).toIntArray()

    private val forwardStore = VariableLengthFloatArray(this.numberOutputRows, this.possibleOutputLengths)
    private val backwardStore = VariableLengthFloatArray(this.numberInputRows, this.possibleInputLengths)

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray {
        this.numberInputColumns = numberInputColumns
        this.numberOutputColumns = this.possibleOutputLengths[computeLengthIndex(numberInputColumns, this.minimumColumns)]

        this.forwardResult = this.forwardStore.get(this.numberOutputColumns)
        this.computeForwardResult(withinBatch, numberInputColumns, input, this.forwardResult, isTraining)

        return this.forwardResult
    }

    protected abstract fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean)

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {
        this.backwardResult = this.backwardStore.get(this.numberInputColumns)
        this.computeBackwardResult(withinBatch, this.numberInputColumns, this.numberOutputColumns, this.forwardResult, chain, this.backwardResult)

        return this.backwardResult
    }

    protected abstract fun computeBackwardResult(withinBatch : Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain : FloatArray, backwardResult: FloatArray)

}