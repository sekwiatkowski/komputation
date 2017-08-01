package shape.komputation.cpu.layers.forward.decoder

import shape.komputation.cpu.functions.add
import shape.komputation.cpu.functions.getStep
import shape.komputation.cpu.functions.setStep
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.SeriesBias
import shape.komputation.cpu.layers.forward.projection.SeriesWeighting
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.matrix.floatZeroColumnVector
import shape.komputation.optimization.Optimizable

// The first input is empty.
// Starting with the second input, the input at step t is the output of step t-1.
class CpuSingleInputDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val hiddenDimension : Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Optimizable {

    private val decoding = FloatArray(this.outputDimension * this.numberSteps)

    override fun forward(withinBatch : Int, encoding: FloatMatrix, isTraining : Boolean): FloatMatrix {

        // Use the encoder output as the first state
        var state = encoding
        // The first input is empty since there is no previous output
        var input = floatZeroColumnVector(this.outputDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            val newState = this.unit.forwardStep(withinBatch, indexStep, state, input, isTraining)

            val output = this.forwardOutput(withinBatch, indexStep, newState, isTraining)

            setStep(this.decoding, indexStep, output.entries, this.outputDimension)

            state = newState

            // The output of step t-1 is the input for step t.
            input = output

        }

        return FloatMatrix(this.outputDimension, this.numberSteps, this.decoding)

    }

    private fun forwardOutput(withinBatch : Int, indexStep: Int, newState: FloatMatrix, isTraining: Boolean): FloatMatrix {

        val weighting = this.weighting.forwardStep(withinBatch, indexStep, newState, isTraining)

        val biased =

            if (this.bias != null) {

                this.bias.forwardStep(weighting)

            }
            else {

                weighting
            }

        val output = this.activations[indexStep].forward(withinBatch, biased, isTraining)

        return output

    }

    private val chainStepEntries = FloatArray(this.outputDimension)

    // Incoming gradient: d chain / d series prediction
    override fun backward(withinBatch: Int, chain: FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        // Differentiate the chain w.r.t. input
        var diffStatePreActivationWrtInput : FloatMatrix? = null

        // Differentiate the chain w.r.t previous state.
        // This is done at each step. For the first step (t=1), the chain is differentiated w.r.t. to the initial state (t=0).
        var diffStatePreActivationWrtPreviousState : FloatMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            getStep(chainEntries, indexStep, this.chainStepEntries, this.outputDimension)

            val diffOutputPreActivationWrtState = backwardOutput(withinBatch, indexStep, this.chainStepEntries, diffStatePreActivationWrtInput?.entries)

            val stateSum = if (diffStatePreActivationWrtPreviousState != null) {

                add(diffStatePreActivationWrtPreviousState.entries, diffOutputPreActivationWrtState.entries, diffOutputPreActivationWrtState.entries, this.hiddenDimension)

                floatColumnVector(*diffOutputPreActivationWrtState.entries)

            }
            else {

                diffOutputPreActivationWrtState

            }

            val (newDiffStatePreActivationWrtPreviousState, newDiffStatePreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, stateSum)

            diffStatePreActivationWrtInput = newDiffStatePreActivationWrtInput
            diffStatePreActivationWrtPreviousState = newDiffStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return diffStatePreActivationWrtPreviousState!!

    }

    private val outputSum = FloatArray(this.outputDimension)

    private fun backwardOutput(withinBatch: Int, indexStep: Int, chainStep: FloatArray, diffStatePreActivationWrtInput: FloatArray?): FloatMatrix {

        // The input gradient for step t+1 is added to the chain step t
        val addition = if (diffStatePreActivationWrtInput != null) {

            // d chain / d output(index+1) * d output(index+1) / d input(index + 1) *  d input(index + 1) / d output(index)
            add(diffStatePreActivationWrtInput, chainStep, this.outputSum, this.outputDimension)

            this.outputSum

        }
        // except in the case of the last step (t = T)
        else {

            chainStep

        }

        val diffOutputWrtOutputPreActivation = this.activations[indexStep].backward(withinBatch, floatColumnVector(*addition))

        this.bias?.backwardStep(diffOutputWrtOutputPreActivation)

        val diffOutputPreActivationWrtState = this.weighting.backwardStep(withinBatch, indexStep, diffOutputWrtOutputPreActivation)

        return diffOutputPreActivationWrtState

    }

    override fun optimize(scalingFactor : Float) {

        if (this.unit is Optimizable) {

            this.unit.optimize(scalingFactor)

        }

        this.weighting.optimize(scalingFactor)
        this.bias?.optimize(scalingFactor)

    }

}