package shape.komputation.cpu.layers.forward.decoder

import shape.komputation.cpu.functions.add
import shape.komputation.cpu.functions.extractStep
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.forward.activation.CpuActivationLayer
import shape.komputation.cpu.layers.forward.projection.SeriesBias
import shape.komputation.cpu.layers.forward.projection.SeriesWeighting
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
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

    override fun forward(input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        val seriesOutput = floatZeroMatrix(this.outputDimension, this.numberSteps)

        // Start with a zero state
        var state = floatZeroColumnVector(this.hiddenDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            // Extract the n-th step input
            val stepInput = input.getColumn(indexStep)

            // Compute the new state
            state = this.unit.forwardStep(indexStep, state, stepInput, isTraining)

            val output = this.forwardOutput(indexStep, state, isTraining)

            // Store the n-th output
            seriesOutput.setColumn(indexStep, output.entries)

        }

        return seriesOutput

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

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: FloatMatrix): FloatMatrix {

        val chainEntries = chain.entries

        val diffStatePreActivationWrtInput = floatZeroMatrix(this.inputDimension, this.numberSteps)
        var diffStatePreActivationWrtPreviousState : FloatMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val chainStep = floatColumnVector(*extractStep(chainEntries, indexStep, outputDimension))

            val diffOutputPreActivationWrtState = this.backwardOutput(indexStep, chainStep)

            val stateSum = if (diffStatePreActivationWrtPreviousState != null) {

                floatColumnVector(*add(diffStatePreActivationWrtPreviousState.entries, diffOutputPreActivationWrtState.entries))

            }
            else {

                diffOutputPreActivationWrtState

            }

            val (newDiffStatePreActivationWrtPreviousState, newDiffStatePreActivationWrtInput) = this.unit.backwardStep(indexStep, stateSum)

            diffStatePreActivationWrtInput.setColumn(indexStep, newDiffStatePreActivationWrtInput.entries)
            diffStatePreActivationWrtPreviousState = newDiffStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return diffStatePreActivationWrtInput

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