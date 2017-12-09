package com.komputation.cpu.layers

fun computeNumberPossibleLengths(minimumLength: Int, maximumLength: Int) =
    maximumLength - minimumLength + 1

fun computeLengthIndex(length : Int, minimumLength: Int) =
    length - minimumLength

fun computePossibleLengths(minimumLength: Int, numberPossibleLengths : Int) =
    IntArray(numberPossibleLengths) { index -> index + minimumLength }

abstract class BaseCpuVariableLengthForwardLayer(
    private val name: String?,
    final override val numberInputRows : Int,
    final override val numberOutputRows : Int,
    protected val minimumColumns : Int,
    protected val maximumColumns : Int) : CpuForwardLayer {

    override var backwardResult = FloatArray(0)
    override var numberInputColumns = -1

    override var forwardResult = FloatArray(0)
    override var numberOutputColumns = -1

    protected abstract fun computeNumberOutputColumns(inputLength : Int) : Int

    protected val numberPossibleLengths = computeNumberPossibleLengths(this.minimumColumns, this.maximumColumns)
    protected val possibleLengths = computePossibleLengths(this.minimumColumns, this.numberPossibleLengths)

    private val forwardStore = VariableLengthFloatArray(this.numberOutputRows, this.minimumColumns, this.possibleLengths, { inputLength -> computeNumberOutputColumns(inputLength) })
    private val backwardStore = VariableLengthFloatArray(this.numberInputRows, this.minimumColumns, this.possibleLengths, { inputLength -> inputLength })

    open fun prepare(numberInputColumns: Int) { }

    override fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray {
        this.prepare(numberInputColumns)

        this.numberInputColumns = numberInputColumns
        this.numberOutputColumns = computeNumberOutputColumns(numberInputColumns)

        this.forwardResult = this.forwardStore.get(numberInputColumns)
        this.computeForwardResult(withinBatch, numberInputColumns, input, isTraining, this.forwardResult)

        return this.forwardResult
    }

    protected abstract fun computeForwardResult(withinBatch : Int, numberInputColumns : Int, input : FloatArray, isTraining : Boolean, forwardResult: FloatArray)

    override fun backward(withinBatch : Int, chain : FloatArray) : FloatArray {
        this.backwardResult = this.backwardStore.get(this.numberInputColumns)
        this.computeBackwardResult(withinBatch, this.numberInputColumns, this.numberOutputColumns, this.forwardResult, chain, this.backwardResult)

        return this.backwardResult
    }

    protected abstract fun computeBackwardResult(withinBatch : Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain : FloatArray, backwardResult: FloatArray)

}