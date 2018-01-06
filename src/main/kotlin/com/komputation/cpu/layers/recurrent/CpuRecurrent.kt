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

fun computePossibleStepsLeftToRight(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex -> 0 until (minimumSteps + lengthIndex) }

fun computePossibleStepsRightToLeft(minimumSteps: Int, numberPossibleLengths : Int) =
    Array(numberPossibleLengths) { lengthIndex -> (minimumSteps + lengthIndex - 1) downTo 0 }

class CpuRecurrent(
    name: String?,
    private val minimumSteps: Int,
    private val maximumSteps: Int,
    private val hiddenDimension: Int,
    private val inputProjection: CpuProjection,
    private val initialState: FloatArray,
    private val previousHiddenStateWeighting: CpuParameterizedSeries,
    private val additions: CpuCombinationSeries,
    private val activation: CpuSeries,
    private val direction: Direction,
    private val resultExtraction: ResultExtractionStrategy) : BaseCpuHigherOrderContinuation(name, inputProjection, resultExtraction), Optimizable {

    private val stepWeightedInput = FloatArray(this.hiddenDimension)

    private val numberPossibleLengths = computeNumberPossibleLengths(this.minimumSteps, this.maximumSteps)
    private val possibleLengths = computePossibleLengths(this.minimumSteps, this.numberPossibleLengths)
    private val possibleStepsLeftToRight = computePossibleStepsLeftToRight(this.minimumSteps, this.numberPossibleLengths)
    private val possibleStepsRightToLeft = computePossibleStepsRightToLeft(this.minimumSteps, this.numberPossibleLengths)

    private val forwardStepsOverPossibleLengths = when (this.direction) {
        Direction.LeftToRight -> this.possibleStepsLeftToRight
        Direction.RightToLeft -> this.possibleStepsRightToLeft
    }

    private val backwardStepsOverPossibleLengths = when (this.direction) {
        Direction.LeftToRight -> this.possibleStepsRightToLeft
        Direction.RightToLeft -> this.possibleStepsLeftToRight
    }

    override fun forward(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {
        val projectedInput = this.inputProjection.forward(withinBatch, numberInputColumns, input, isTraining)

        var previousHiddenState = this.initialState

        val steps = this.forwardStepsOverPossibleLengths[computeLengthIndex(this.numberInputColumns, this.minimumSteps)]

        for (step in steps) {
            getColumn(projectedInput, step, this.hiddenDimension, this.stepWeightedInput)

            val weightedPreviousHiddenState = this.previousHiddenStateWeighting.forwardStep(withinBatch, step, 1, previousHiddenState, isTraining)

            val preActivation = this.additions.forwardStep(step, this.stepWeightedInput, weightedPreviousHiddenState, 1)

            val hiddenState = this.activation.forwardStep(withinBatch, step, 1, preActivation, isTraining)

            previousHiddenState = hiddenState
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

        val steps = this.backwardStepsOverPossibleLengths[computeLengthIndex(this.numberInputColumns, this.minimumSteps)]

        for (step in steps) {
            val stepChain = this.resultExtraction.backwardStep(step, chain, previousBackwardPreviousHiddenState)

            // dh_t / d(Wx_t + Uh_(t-1) + b) = df(Wx_t + Uh_(t-1) + b) / d(Wx_t + Uh_(t-1) + b)
            val stepBackwardPreActivation = this.activation.backwardStep(withinBatch, step, stepChain)

            // d(Wx_t + Uh_(t-1) + b) / dUh_(t-1)
            val backwardPreviousHiddenState = this.previousHiddenStateWeighting.backwardStep(withinBatch, step, stepBackwardPreActivation)
            previousBackwardPreviousHiddenState = backwardPreviousHiddenState

            setColumn(stepBackwardPreActivation, step, this.hiddenDimension, backwardPreActivation)
        }

        this.previousHiddenStateWeighting.backwardSeries()

        // d(Wx_t + Uh_(t-1) + b) / dW
        // d(Wx_t + Uh_(t-1) + b) / dx_t
        // d(Wx_t + Uh_(t-1) + b) / db
        return this.inputProjection.backward(withinBatch, backwardPreActivation)
    }

    override fun optimize(batchSize: Int) {
        this.inputProjection.optimize(batchSize)
        this.previousHiddenStateWeighting.optimize(batchSize)
    }

}