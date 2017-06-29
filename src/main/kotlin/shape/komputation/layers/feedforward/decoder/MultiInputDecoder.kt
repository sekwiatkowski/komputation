package shape.komputation.layers.feedforward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.add
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.createActivationLayers
import shape.komputation.layers.feedforward.projection.SeriesBias
import shape.komputation.layers.feedforward.projection.SeriesWeighting
import shape.komputation.layers.feedforward.projection.createSeriesBias
import shape.komputation.layers.feedforward.projection.createSeriesWeighting
import shape.komputation.layers.feedforward.units.RecurrentUnit
import shape.komputation.matrix.*
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class MultiInputDecoder(
    name : String?,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<ActivationLayer>) : ContinuationLayer(name), Optimizable {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        input as SequenceMatrix

        val seriesOutput = zeroSequenceMatrix(this.numberSteps, this.outputDimension)

        // Start with a zero state
        var state = doubleZeroColumnVector(this.hiddenDimension)

        for (indexStep in 0..this.numberSteps - 1) {

            // Extract the n-th step input
            val stepInput = input.getStep(indexStep)

            // Compute the new state
            state = this.unit.forwardStep(indexStep, state, stepInput)

            val output = this.forwardOutput(indexStep, state)

            // Store the n-th output
            seriesOutput.setStep(indexStep, output.entries)

        }

        return seriesOutput

    }

    private fun forwardOutput(indexStep: Int, state: DoubleMatrix): DoubleMatrix {

        val weighting = this.weighting.forwardStep(indexStep, state)

        val biased =

            if (this.bias != null) {

                bias.forwardStep(weighting)

            }
            else {

                weighting
            }

        val output = this.activations[indexStep].forward(biased)

        return output

    }

    // Incoming gradient: d chain / d series prediction
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val diffStatePreActivationWrtInput = zeroSequenceMatrix(this.numberSteps, this.inputDimension)
        var diffStatePreActivationWrtPreviousState : DoubleMatrix? = null

        for (indexStep in this.numberSteps - 1 downTo 0) {

            val chainStep = doubleColumnVector(*extractStep(chainEntries, indexStep, outputDimension))

            val diffOutputPreActivationWrtState = this.backwardOutput(indexStep, chainStep)

            val stateSum = if (diffStatePreActivationWrtPreviousState != null) {

                doubleColumnVector(*add(diffStatePreActivationWrtPreviousState.entries, diffOutputPreActivationWrtState.entries))

            }
            else {

                diffOutputPreActivationWrtState

            }

            val (newDiffStatePreActivationWrtPreviousState, newDiffStatePreActivationWrtInput) = this.unit.backwardStep(indexStep, stateSum)

            diffStatePreActivationWrtInput.setStep(indexStep, newDiffStatePreActivationWrtInput.entries)
            diffStatePreActivationWrtPreviousState = newDiffStatePreActivationWrtPreviousState

        }

        this.unit.backwardSeries()

        this.weighting.backwardSeries()
        this.bias?.backwardSeries()

        return diffStatePreActivationWrtInput

    }

    private fun backwardOutput(indexStep: Int, chainStep: DoubleMatrix): DoubleMatrix {

        val diffOutputWrtOutputPreActivation = this.activations[indexStep].backward(chainStep)

        val diffOutputPreActivationWrtState = this.weighting.backwardStep(indexStep, diffOutputWrtOutputPreActivation)

        this.bias?.backwardStep(diffOutputWrtOutputPreActivation)

        return diffOutputPreActivationWrtState

    }

    override fun optimize() {

        if (this.unit is Optimizable) {

            this.unit.optimize()

        }

        this.weighting.optimize()
        this.bias?.optimize()

    }

}

fun createMultiInputDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    createMultiInputDecoder(
        null,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)

fun createMultiInputDecoder(
    name : String?,
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?): MultiInputDecoder {

    val weightingSeriesName = concatenateNames(name, "weighting")
    val weightingStepName = concatenateNames(name, "weighting-step")
    val weighting = createSeriesWeighting(weightingSeriesName, weightingStepName, numberSteps, false, hiddenDimension, outputDimension, weightInitializationStrategy, optimizationStrategy)

    val bias =
        if (biasInitializationStrategy != null) {

            val biasSeriesName = concatenateNames(name, "bias")
            createSeriesBias(biasSeriesName, outputDimension, biasInitializationStrategy, optimizationStrategy)

        }
        else {

            null

        }

    val activationName = concatenateNames(name, "activation")
    val activations = createActivationLayers(
        numberSteps,
        activationName,
        activationFunction
    )

    return MultiInputDecoder(
        name,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        unit,
        weighting,
        bias,
        activations)

}