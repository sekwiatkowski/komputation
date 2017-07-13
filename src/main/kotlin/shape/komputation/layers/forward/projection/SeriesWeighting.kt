package shape.komputation.layers.forward.projection

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.identityLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class SeriesWeighting internal constructor(
    private val name : String?,
    private val weightings: Array<BaseForwardLayer>,
    private val weights: DoubleArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule?) {

    private val numberWeightEntries = weights.size

    fun forwardStep(step : Int, input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        return this.weightings[step].forward(input, isTraining)

    }

    fun backwardStep(step: Int, chain: DoubleMatrix) : DoubleMatrix {

        val backward = this.weightings[step].backward(chain)

        return backward

    }

    fun backwardSeries() {

        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()

    }

    fun optimize(scalingFactor : Double) {

        if (this.updateRule != null) {

            updateDensely(this.weights, this.batchAccumulator.getAccumulation(), scalingFactor, this.updateRule)

        }

        this.batchAccumulator.reset()

    }

}

fun seriesWeighting(
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    inputDimension: Int,
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationStrategy?) =

    seriesWeighting(
        null,
        null,
        numberSteps,
        useIdentityAtFirstStep,
        inputDimension,
        outputDimension,
        initializationStrategy,
        optimizationStrategy
    )

fun seriesWeighting(
    seriesName: String?,
    stepName: String?,
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    inputDimension: Int,
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationStrategy?) : SeriesWeighting {

    val weights = initializeWeights(initializationStrategy, outputDimension, inputDimension, inputDimension)

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weightUpdateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val numberEntries = inputDimension * outputDimension
    val seriesAccumulator = DenseAccumulator(numberEntries)
    val batchAccumulator = DenseAccumulator(numberEntries)

    val stepWeightings = Array(numberSteps) { indexStep ->

        val stepProjectionName = concatenateNames(stepName, indexStep.toString())

        if (useIdentityAtFirstStep && indexStep == 0) {

            identityLayer(stepProjectionName).buildForCpu()

        }
        else {

            stepWeighting(stepProjectionName, numberWeightRows, numberWeightColumns, weights, seriesAccumulator, weightUpdateRule)
        }

    }

    val updateRule = optimizationStrategy?.invoke(inputDimension, outputDimension)

    val seriesWeighting = SeriesWeighting(
        seriesName,
        stepWeightings,
        weights,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

    return seriesWeighting

}

private fun stepWeighting(
    name : String?,
    numberWeightRows: Int,
    numberWeightColumns: Int,
    weights : DoubleArray,
    weightAccumulator: DenseAccumulator,
    weightUpdateRule : UpdateRule? = null) =

    CpuProjectionLayer(name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule)