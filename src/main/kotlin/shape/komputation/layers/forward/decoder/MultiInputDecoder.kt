package shape.komputation.layers.forward.decoder

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.add
import shape.komputation.functions.extractStep
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.layers.forward.activation.activationLayers
import shape.komputation.layers.forward.projection.SeriesBias
import shape.komputation.layers.forward.projection.SeriesWeighting
import shape.komputation.layers.forward.projection.seriesBias
import shape.komputation.layers.forward.projection.seriesWeighting
import shape.komputation.layers.forward.units.RecurrentUnit
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleZeroColumnVector
import shape.komputation.matrix.doubleZeroMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class MultiInputDecoder internal constructor(
    name : String?,
    private val numberSteps : Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension : Int,
    private val unit : RecurrentUnit,
    private val weighting: SeriesWeighting,
    private val bias: SeriesBias?,
    private val activations: Array<ActivationLayer>) : ForwardLayer(name), Optimizable {

    override fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val seriesOutput = doubleZeroMatrix(this.outputDimension, this.numberSteps)

        // Start with a zero state
        var state = doubleZeroColumnVector(this.hiddenDimension)

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

    private fun forwardOutput(indexStep: Int, state: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

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
    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val chainEntries = chain.entries

        val diffStatePreActivationWrtInput = doubleZeroMatrix(this.inputDimension, this.numberSteps)
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

            diffStatePreActivationWrtInput.setColumn(indexStep, newDiffStatePreActivationWrtInput.entries)
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

fun multiInputDecoder(
    numberSteps: Int,
    inputDimension: Int,
    hiddenDimension: Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    multiInputDecoder(
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

fun multiInputDecoder(
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
    val weighting = seriesWeighting(weightingSeriesName, weightingStepName, numberSteps, false, hiddenDimension, outputDimension, weightInitializationStrategy, optimizationStrategy)

    val bias =
        if (biasInitializationStrategy != null) {

            val biasSeriesName = concatenateNames(name, "bias")
            seriesBias(biasSeriesName, outputDimension, biasInitializationStrategy, optimizationStrategy)

        }
        else {

            null

        }

    val activationName = concatenateNames(name, "activation")
    val activations = activationLayers(
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