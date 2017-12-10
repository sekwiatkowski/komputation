package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.functions.getColumn
import com.komputation.cpu.functions.setColumn
import com.komputation.cpu.layers.BaseCpuHigherOrderLayer
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.cpu.layers.computeNumberPossibleLengths
import com.komputation.cpu.layers.computePossibleLengths
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.cpu.layers.recurrent.extraction.ResultExtractionStrategy
import com.komputation.cpu.layers.recurrent.series.ParameterizedSeries
import com.komputation.cpu.layers.recurrent.series.Series
import com.komputation.optimization.Optimizable

class CpuRecurrentLayer(
    name : String?,
    private val minimumSteps : Int,
    private val maximumSteps : Int,
    private val hiddenDimension : Int,
    private val inputWeighting : CpuWeightingLayer,
    private val initialState : FloatArray,
    private val previousHiddenStateWeighting: ParameterizedSeries,
    private val additions : Array<CpuAdditionCombination>,
    private val bias: ParameterizedSeries?,
    private val activation: Series,
    private val resultExtraction: ResultExtractionStrategy) : BaseCpuHigherOrderLayer(name, inputWeighting, resultExtraction), Optimizable {

    private val stepWeightedInput = FloatArray(this.hiddenDimension)


    override fun forward(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {
        val weightedInput = this.inputWeighting.forward(withinBatch, numberInputColumns, input, isTraining)

        var previousHiddenState = this.initialState

        for (step in 0 until numberInputColumns) {
            getColumn(weightedInput, step, this.hiddenDimension, this.stepWeightedInput)

            val weightedPreviousHiddenState = this.previousHiddenStateWeighting.forwardStep(withinBatch, step, 1, previousHiddenState, isTraining)

            val addition = this.additions[step].forward(this.stepWeightedInput, weightedPreviousHiddenState)

            val finalPreActivation =
                if(this.bias != null)
                    this.bias.forwardStep(withinBatch, step, 1, addition, isTraining)
                else
                    addition

            val hiddenState = this.activation.forwardStep(withinBatch, step, 1, finalPreActivation, isTraining)

            previousHiddenState = hiddenState
        }

        return this.resultExtraction.extractResult(numberInputColumns)
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
    protected val numberPossibleLengths = computeNumberPossibleLengths(this.minimumSteps, this.maximumSteps)
    protected val possibleLengths = computePossibleLengths(this.minimumSteps, this.numberPossibleLengths)
    private val backwardStore = VariableLengthFloatArray(this.hiddenDimension, this.possibleLengths)

    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {
        val backwardPreActivation = this.backwardStore.get(this.numberInputColumns)

        var previousBackwardPreviousHiddenState : FloatArray? = null

        val lastStep = this.numberInputColumns - 1

        for (step in lastStep downTo 0) {

            val stepChain = this.resultExtraction.backwardStep(step, chain, previousBackwardPreviousHiddenState)

            // dh_t / d(Wx_t + Uh_(t-1) + b) = df(Wx_t + Uh_(t-1) + b) / d(Wx_t + Uh_(t-1) + b)
            val stepBackwardPreActivation = this.activation.backwardStep(withinBatch, step, stepChain)

            // d(Wx_t + Uh_(t-1) + b) / dUh_(t-1)
            val backwardPreviousHiddenState = this.previousHiddenStateWeighting.backwardStep(withinBatch, step, stepBackwardPreActivation)
            previousBackwardPreviousHiddenState = backwardPreviousHiddenState

            // d(Wx_t + Uh_(t-1) + b) / db
            this.bias?.backwardStep(withinBatch, step, stepBackwardPreActivation)

            setColumn(stepBackwardPreActivation, step, this.hiddenDimension, backwardPreActivation)
        }

        this.previousHiddenStateWeighting.backwardSeries()
        this.bias?.backwardSeries()

        return this.inputWeighting.backward(withinBatch, backwardPreActivation)
    }

    override fun optimize(batchSize: Int) {
        this.inputWeighting.optimize(batchSize)
        this.previousHiddenStateWeighting.optimize(batchSize)
        this.bias?.optimize(batchSize)
    }

}