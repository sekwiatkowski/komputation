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

class CpuMultiInputDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<CpuActivationLayer>) : BaseCpuForwardLayer(name), Optimizable {

    private val states = Array(this.numberSteps+1) { floatZeroColumnVector(hiddenDimension) }
    private val steps = Array(this.numberSteps) { FloatArray(this.inputDimension) }
    private val forwardEntries = FloatArray(this.outputDimension * this.numberSteps)

    private val backwardEntries = FloatArray(this.inputDimension * this.numberSteps)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        for (indexStep in 0..this.numberSteps - 1) {

            val step = this.steps[indexStep]
            getStep(input.entries, indexStep, step, this.inputDimension)

            // Compute the new state
            val newState = this.unit.forwardStep(withinBatch, indexStep, this.states[indexStep], floatColumnVector(*step), isTraining)
            this.states[indexStep+1] = newState

            val output = this.forwardOutput(withinBatch, indexStep, newState, isTraining)

            // Store the n-th output
            setStep(output.entries, indexStep, this.forwardEntries, this.outputDimension)

        }

        return FloatMatrix(this.outputDimension, this.numberSteps, this.forwardEntries)

    }

    private fun forwardOutput(withinBatch : Int, indexStep: Int, state: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val weighting = this.weighting.forwardStep(withinBatch, indexStep, state, isTraining)

        val biased =

            if (this.bias != null) {

                this.bias.forwardStep(withinBatch, indexStep, weighting, isTraining)

            }
            else {

                weighting
            }

        val output = this.activations[indexStep].forward(withinBatch, biased, isTraining)

        return output

    }

    private val chainStepEntries = FloatArray(this.hiddenDimension)
    private val stateSumEntries = FloatArray(this.hiddenDimension)

    // Incoming gradient: d chain / d series prediction
    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        var diffStatePreActivationWrtPreviousState : FloatMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            getStep(chainEntries, indexStep, this.chainStepEntries, this.outputDimension)
            val chainStep = floatColumnVector(*this.chainStepEntries)

            val diffOutputPreActivationWrtState = this.backwardOutput(withinBatch, indexStep, chainStep)

            val stateSum = if (diffStatePreActivationWrtPreviousState != null) {

                add(diffStatePreActivationWrtPreviousState.entries, diffOutputPreActivationWrtState.entries, this.stateSumEntries, this.hiddenDimension)

                floatColumnVector(*stateSumEntries)

            }
            else {

                diffOutputPreActivationWrtState

            }

            val (newDiffStatePreActivationWrtPreviousState, newDiffStatePreActivationWrtInput) = this.unit.backwardStep(withinBatch, indexStep, stateSum)

            setStep(newDiffStatePreActivationWrtInput.entries, indexStep, this.backwardEntries, this.inputDimension)
            diffStatePreActivationWrtPreviousState = newDiffStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return FloatMatrix(this.inputDimension, this.numberSteps, this.backwardEntries)

    }

    private fun backwardOutput(withinBatch : Int, indexStep: Int, chainStep: FloatMatrix): FloatMatrix {

        val diffOutputWrtOutputPreActivation = this.activations[indexStep].backward(withinBatch, chainStep)

        val diffOutputPreActivationWrtState = this.weighting.backwardStep(withinBatch, indexStep, diffOutputWrtOutputPreActivation)

        this.bias?.backwardStep(withinBatch, indexStep, diffOutputWrtOutputPreActivation)

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