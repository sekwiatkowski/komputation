package com.komputation.cpu.layers.recurrent

import com.komputation.cpu.functions.add
import com.komputation.cpu.functions.getColumn
import com.komputation.cpu.functions.setColumn
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.layers.combination.CpuAdditionCombination
import com.komputation.cpu.layers.forward.projection.CpuWeightingLayer
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuRecurrentLayer(
    name : String?,
    minimumSteps : Int,
    maximumSteps : Int,
    private val hiddenDimension : Int,
    private val inputWeighting : CpuWeightingLayer,
    private val initialState : FloatArray,
    private val previousHiddenStateWeighting: ParameterizedSeries,
    private val additions : Array<CpuAdditionCombination>,
    private val bias: ParameterizedSeries?,
    private val activation: Series) : BaseCpuVariableLengthForwardLayer(name, hiddenDimension, hiddenDimension, minimumSteps, maximumSteps), Resourceful, Optimizable {

    private val stepWeightedInput = FloatArray(this.hiddenDimension)
    private val stepChain = FloatArray(this.hiddenDimension)

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.inputWeighting.acquire(maximumBatchSize)
        this.previousHiddenStateWeighting.acquire(maximumBatchSize)
        this.bias?.acquire(maximumBatchSize)
    }

    override fun release() {
        super.release()

        this.inputWeighting.release()
        this.previousHiddenStateWeighting.release()
        this.bias?.release()
    }

    override fun computeNumberOutputColumns(lengthIndex: Int, length: Int) =
        length

    // h_t = f(Uh + Wx + b)
    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
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

            setColumn(hiddenState, step, this.hiddenDimension, forwardResult)

            previousHiddenState = hiddenState
        }

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
    override fun computeBackwardResult(withinBatch: Int, forwardResult: FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        var previousBackwardPreviousHiddenState : FloatArray? = null

        val lastStep = this.numberInputColumns - 1

        for (step in lastStep downTo 0) {
            getColumn(chain, step, this.hiddenDimension, this.stepChain)

            if(step < lastStep) {
                add(this.stepChain, previousBackwardPreviousHiddenState!!, this.stepChain, this.hiddenDimension)
            }

            // dh_t / d(Wx_t + Uh_(t-1) + b) = df(Wx_t + Uh_(t-1) + b) / d(Wx_t + Uh_(t-1) + b)
            val stepBackwardPreActivation = this.activation.backwardStep(withinBatch, step, this.stepChain)

            // d(Wx_t + Uh_(t-1) + b) / dWx_t
            setColumn(stepBackwardPreActivation, step, this.hiddenDimension, backwardResult)

            // d(Wx_t + Uh_(t-1) + b) / dUh_(t-1)
            val backwardPreviousHiddenState = this.previousHiddenStateWeighting.backwardStep(withinBatch, step, stepBackwardPreActivation)
            previousBackwardPreviousHiddenState = backwardPreviousHiddenState

            // d(Wx_t + Uh_(t-1) + b) / db
            this.bias?.backwardStep(withinBatch, step, backwardResult)
        }

        this.previousHiddenStateWeighting.backwardSeries()
        this.bias?.backwardSeries()

        this.inputWeighting.backward(withinBatch, backwardResult)
    }

    override fun optimize(batchSize: Int) {
        this.inputWeighting.optimize(batchSize)
        this.previousHiddenStateWeighting.optimize(batchSize)
        this.bias?.optimize(batchSize)
    }

}