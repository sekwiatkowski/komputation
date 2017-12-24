package com.komputation.cuda.layers.recurrent


/* class CudaRecurrentLayer(
    private val name : String?,
    private val minimumLength : Int,
    private val maximumLength : Int,
    private val inputWeighting : CublasWeightingLayer,
    private val bias : CudaBiasLayer?,
    private val activation : ActivationFunction,
    private val direction: Direction) : BaseCudaForwardLayer(name), Resourceful {

    override val deviceForwardResult: Pointer
        get() = this.deviceResult
    override val numberOutputRows: Int
        get() = this.numberInputRows
    override val maximumOutputColumns: Int
        get() = this.maximumInputColumns
    override val deviceBackwardResult: Pointer
        get() = this.inputWeighting.deviceBackwardResult
    override val numberInputRows: Int
        get() = this.inputWeighting.numberInputRows
    override val maximumInputColumns: Int
        get() = this.inputWeighting.maximumInputColumns

    private val deviceResult = Pointer()
    private val pointerToResult = Pointer.to(this.deviceResult)

    private val hiddenDimension = this.inputWeighting.numberOutputRows

    override fun acquire(maximumBatchSize: Int) {
        acquire(maximumBatchSize * this.hiddenDimension * this.maximumLength)
    }

    override fun release() {
        cudaFree(this.deviceResult)
    }

    private val numberPossibleLengths = computeNumberPossibleLengths(this.minimumLength, this.maximumLength)
    private val possibleStepsLeftToRight = computePossibleStepsLeftToRight(this.minimumLength, this.numberPossibleLengths)
    private val possibleStepsRightToLeft = computePossibleStepsRightToLeft(this.minimumLength, this.numberPossibleLengths)

    private val forwardStepsOverPossibleLengths = when (this.direction) {
        Direction.leftToRight -> this.possibleStepsLeftToRight
        Direction.rightToLeft -> this.possibleStepsRightToLeft
    }

    private val backwardStepsOverPossibleLengths = when (this.direction) {
        Direction.leftToRight -> this.possibleStepsRightToLeft
        Direction.rightToLeft -> this.possibleStepsLeftToRight
    }

    override fun forward(batchSize: Int, deviceInput: Pointer, batchMaximumInputColumns : Int, isTraining: Boolean): Pointer {

        val deviceWeightedInput = this.inputWeighting.forward(batchSize, deviceInput, batchMaximumInputColumns, isTraining)

        val finalInput = if (this.bias != null) {
            this.bias.forward(batchSize, deviceWeightedInput, 1, isTraining)
        }
        else {
            deviceWeightedInput
        }

        /* val steps = this.forwardStepsOverPossibleLengths[computeLengthIndex(this.numberInputColumns, this.minimumLength)]

        for (step in steps) {

        } */

        return deviceResult

    }

    override fun backward(batchSize: Int, chain: Pointer): Pointer {
        TODO("not implemented")
    }

} */