package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.functions.getColumn
import com.komputation.cpu.functions.setColumn
import com.komputation.cpu.layers.*
import com.komputation.cpu.layers.continuation.projection.CpuProjection
import com.komputation.cpu.layers.recurrent.extraction.ResultExtractionStrategy
import com.komputation.cpu.layers.recurrent.series.CpuCombinationSeries
import com.komputation.cpu.layers.recurrent.series.CpuParameterizedSeries
import com.komputation.cpu.layers.recurrent.series.CpuSeries
import com.komputation.optimization.Optimizable

fun computePossibleForwardStepsLeftToRight(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex ->
        Pair(0, Array(minimumSteps + lengthIndex - 1) { index -> index + 1 })
    }

fun computePossibleForwardStepsRightToLeft(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex ->
        val numberSteps = minimumSteps + lengthIndex
        val numberRemainingSteps = numberSteps-1

        Pair(numberRemainingSteps, Array(numberRemainingSteps) { index -> numberRemainingSteps - index - 1 })
    }

fun computePossibleBackwardStepsLeftToRight(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex ->
        val numberSteps = minimumSteps + lengthIndex

        Pair(Array(minimumSteps + lengthIndex - 1) { index -> numberSteps - index - 1 }, 0)
    }

fun computePossibleBackwardStepsRightToLeft(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex ->
        val numberSteps = minimumSteps + lengthIndex
        val numberRemainingSteps = numberSteps-1

        Pair(Array(numberRemainingSteps) { index -> index }, numberSteps - 1)
    }

class CpuRecurrent(
    name: String?,
    private val minimumSteps: Int,
    private val maximumSteps: Int,
    private val hiddenDimension: Int,
    private val inputProjection: CpuProjection,
    private val previousStateWeighting: CpuParameterizedSeries,
    private val additions: CpuCombinationSeries,
    private val activation: CpuSeries,
    private val direction: Direction,
    private val resultExtraction: ResultExtractionStrategy) : BaseCpuHigherOrderContinuation(name, inputProjection, resultExtraction), Optimizable {

    private val stepWeightedInput = FloatArray(this.hiddenDimension)

    private val numberPossibleLengths = computeNumberPossibleLengths(this.minimumSteps, this.maximumSteps)
    private val possibleLengths = computePossibleLengths(this.minimumSteps, this.numberPossibleLengths)

    private val possibleForwardStepsLeftToRight = computePossibleForwardStepsLeftToRight(this.minimumSteps, this.numberPossibleLengths)
    private val possibleForwardStepsRightToLeft = computePossibleForwardStepsRightToLeft(this.minimumSteps, this.numberPossibleLengths)

    private val possibleBackwardStepsLeftToRight = computePossibleBackwardStepsLeftToRight(this.minimumSteps, this.numberPossibleLengths)
    private val possibleBackwardStepsRightToLeft = computePossibleBackwardStepsRightToLeft(this.minimumSteps, this.numberPossibleLengths)

    private val possibleForwardSteps = when (this.direction) {
        Direction.LeftToRight -> this.possibleForwardStepsLeftToRight
        Direction.RightToLeft -> this.possibleForwardStepsRightToLeft
    }
    private val possibleBackwardSteps = when (this.direction) {
        Direction.LeftToRight -> this.possibleBackwardStepsLeftToRight
        Direction.RightToLeft -> this.possibleBackwardStepsRightToLeft
    }

    // right to left:
    // 4 | 3 2 1 0
    //           ^ position of first previous state weighting

    // left to right:
    // 0 | 1 2 3 4
    //     ^ position of first previous state weighting
    private val positionOfFristPreviousStateWeighting = when (this.direction) {
        Direction.LeftToRight -> 1
        Direction.RightToLeft -> 0
    }

    override fun forward(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {
        val projectedInput = this.inputProjection.forward(withinBatch, numberInputColumns, input, isTraining)

        val (firstInputStep, remainingInputSteps) = this.possibleForwardSteps[computeLengthIndex(this.numberInputColumns, this.minimumSteps)]

        // First step:
        getColumn(projectedInput, firstInputStep, this.hiddenDimension, this.stepWeightedInput)
        var previousHiddenState = this.activation.forwardStep(withinBatch, firstInputStep, 1, this.stepWeightedInput, isTraining)

        // Remaining steps:
        for(indexRemainingStep in 0 until numberInputColumns-1) {
            val remainingInputStep = remainingInputSteps[indexRemainingStep]
            getColumn(projectedInput, remainingInputStep, this.hiddenDimension, this.stepWeightedInput)

            val weightedPreviousHiddenState = this.previousStateWeighting.forwardStep(withinBatch, remainingInputStep-this.positionOfFristPreviousStateWeighting, 1, previousHiddenState, isTraining)
            val preActivation = this.additions.forwardStep(remainingInputStep- positionOfFristPreviousStateWeighting, this.stepWeightedInput, weightedPreviousHiddenState, 1)

            previousHiddenState = this.activation.forwardStep(withinBatch, remainingInputStep, 1, preActivation, isTraining)
        }

        return this.resultExtraction.extractResult(this.activation, numberInputColumns)
    }

    /*
          y1    y2           yT
          |     |            |
    h0 -> h1 -> h2 -> ... -> hT
          |     |            |
          p1    p2           pT

      dy_2/dWx_2 + dh_3/dWx_2
    = dy_2/dh_2 * dh_2/dWx_2 + dh_3/dWx_2 * dh_2/dWx_2
    = [ dy_2/dh_2 * df(Uh_1+Wx_2)/dWx_2 ] +
                   =dh2
      [ df(Uh_2+Wx_3)/df(Uh_1+Wx_2) * df(Uh_1+Wx_2)/dWx_2 ]
        =dh3          =dh2            =dh2
    = [ dy_2/df(Uh_1+Wx_2) * df(Uh_1+Wx_2)/d(Uh_1+Wx_2) * dUh_1+Wx_2/dWx_2 ] +
                            =dh2
      [ df(Uh_2+Wx_3)/df(Uh_1+Wx_2) * df(Uh_1+Wx_2)/d(Uh_1+Wx_2) * dUh_1+Wx_2/dWx_2 ]
        =dh3          =dh2            =dh2

    = [ dy_2/df(Uh_1+Wx_2) + df(Uh_2+Wx_3)/df(Uh_1+Wx_2) ] * df(Uh_1+Wx_2)/d(Uh_1+Wx_2) * dUh_1+Wx_2/dWx_2
    */
    private val backwardStore = VariableLengthFloatArray(this.hiddenDimension, this.possibleLengths)

    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {
        val backwardPreActivation = this.backwardStore.get(this.numberInputColumns)

        var previousBackwardPreviousHiddenState : FloatArray? = null

        val (initialInputSteps, lastInputStep) = this.possibleBackwardSteps[computeLengthIndex(this.numberInputColumns, this.minimumSteps)]

        // Initial steps
        for(initialInputIndex in 0 until this.numberInputColumns-1) {

            val initialInputStep = initialInputSteps[initialInputIndex]

            val stepChain = this.resultExtraction.backwardStep(initialInputStep, chain, previousBackwardPreviousHiddenState)

            // dh_t / d(Wx_t + Uh_(t-1) + b) = df(Wx_t + Uh_(t-1) + b) / d(Wx_t + Uh_(t-1) + b)
            val stepBackwardPreActivation = this.activation.backwardStep(withinBatch, initialInputStep, stepChain)

            // d(Wx_t + Uh_(t-1) + b) / dUh_(t-1)
            val backwardPreviousHiddenState = this.previousStateWeighting.backwardStep(withinBatch, initialInputStep-this.positionOfFristPreviousStateWeighting, stepBackwardPreActivation)
            previousBackwardPreviousHiddenState = backwardPreviousHiddenState

            setColumn(stepBackwardPreActivation, initialInputStep, this.hiddenDimension, backwardPreActivation)
        }

        // Last steps
        val stepChain = this.resultExtraction.backwardStep(lastInputStep, chain, previousBackwardPreviousHiddenState)
        val stepBackwardPreActivation = this.activation.backwardStep(withinBatch, lastInputStep, stepChain)
        setColumn(stepBackwardPreActivation, lastInputStep, this.hiddenDimension, backwardPreActivation)

        this.previousStateWeighting.backwardSeries()

        // d(Wx_t + Uh_(t-1) + b) / dW
        // d(Wx_t + Uh_(t-1) + b) / dx_t
        // d(Wx_t + Uh_(t-1) + b) / db
        val backwardResult = this.inputProjection.backward(withinBatch, backwardPreActivation)

        return backwardResult
    }

    override fun optimize(batchSize: Int) {
        this.inputProjection.optimize(batchSize)
        this.previousStateWeighting.optimize(batchSize)
    }

}