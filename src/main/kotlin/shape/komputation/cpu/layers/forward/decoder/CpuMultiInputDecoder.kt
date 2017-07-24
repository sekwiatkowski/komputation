package shape.komputation.cpu.layers.forward.decoder

import shape.komputation.cpu.functions.add
import shape.komputation.cpu.functions.getStep
import shape.komputation.cpu.functions.setStep
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.SeriesBias
import shape.komputation.cpu.layers.forward.projection.SeriesWeighting
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.matrix.floatZeroColumnVector
import shape.komputation.matrix.floatZeroMatrix
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

    private val forwardEntries = FloatArray(this.outputDimension * this.numberSteps)
    private val backwardEntries = FloatArray(this.inputDimension * this.numberSteps)

    private val inputStepEntries = FloatArray(this.inputDimension)
    private val inputStep = FloatMatrix(this.inputDimension, 1, this.inputStepEntries)

    override fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        // Start with a zero state
        var state = floatZeroColumnVector(this.hiddenDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            // Extract the n-th step input
            getStep(input.entries, indexStep, this.inputStepEntries, this.inputDimension)

            // Compute the new state
            state = this.unit.forwardStep(indexStep, state, inputStep, isTraining)

            val output = this.forwardOutput(indexStep, state, isTraining)

            // Store the n-th output
            setStep(this.forwardEntries, indexStep, output.entries, this.outputDimension)

        }

        return FloatMatrix(this.outputDimension, this.numberSteps, this.forwardEntries)

    }

    private fun forwardOutput(indexStep: Int, state: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val weighting = this.weighting.forwardStep(indexStep, state, isTraining)

        val biased =

            if (this.bias != null) {

                bias.forwardStep(weighting)

            }
            else {

                weighting
            }

        val output = this.activations[indexStep].forward(biased, isTraining)

        return output

    }

    private val chainStepEntries = FloatArray(this.hiddenDimension)
    private val stateSumEntries = FloatArray(this.hiddenDimension)

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        var diffStatePreActivationWrtPreviousState : FloatMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            getStep(chainEntries, indexStep, this.chainStepEntries, this.outputDimension)
            val chainStep = floatColumnVector(*this.chainStepEntries)

            val diffOutputPreActivationWrtState = this.backwardOutput(indexStep, chainStep)

            val stateSum = if (diffStatePreActivationWrtPreviousState != null) {

                add(diffStatePreActivationWrtPreviousState.entries, diffOutputPreActivationWrtState.entries, this.stateSumEntries, this.hiddenDimension)

                floatColumnVector(*stateSumEntries)

            }
            else {

                diffOutputPreActivationWrtState

            }

            val (newDiffStatePreActivationWrtPreviousState, newDiffStatePreActivationWrtInput) = this.unit.backwardStep(indexStep, stateSum)

            setStep(this.backwardEntries, indexStep, newDiffStatePreActivationWrtInput.entries, this.inputDimension)
            diffStatePreActivationWrtPreviousState = newDiffStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return FloatMatrix(this.inputDimension, this.numberSteps, this.backwardEntries)

    }

    private fun backwardOutput(indexStep: Int, chainStep: FloatMatrix): FloatMatrix {

        val diffOutputWrtOutputPreActivation = this.activations[indexStep].backward(chainStep)

        val diffOutputPreActivationWrtState = this.weighting.backwardStep(indexStep, diffOutputWrtOutputPreActivation)

        this.bias?.backwardStep(diffOutputWrtOutputPreActivation)

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