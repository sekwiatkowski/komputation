package shape.komputation.layers.feedforward.projection

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.createIdentityLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class SeriesWeighting(
    private val name : String?,
    private val weightings: Array<ContinuationLayer>,
    private val weights: DoubleArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule?) {

    private val numberWeightEntries = weights.size

    fun forwardStep(step : Int, input: DoubleMatrix): DoubleMatrix {

        return weightings[step].forward(input)

    }

    fun backwardStep(step: Int, chain: DoubleMatrix) : DoubleMatrix {

        val backward = weightings[step].backward(chain)

        return backward

    }

    fun backwardSeries() {

        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()

    }

    fun optimize() {

        if (this.updateRule != null) {

            updateDensely(this.weights, this.numberWeightEntries, this.batchAccumulator.getAccumulation(), this.batchAccumulator.getCount(), this.updateRule)

        }

        this.batchAccumulator.reset()

    }

}

fun createSeriesWeighting(
    seriesName: String?,
    stepName: String?,
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    inputDimension: Int,
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationStrategy?) : SeriesWeighting {

    val weights = initializeMatrix(initializationStrategy, outputDimension, inputDimension)

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weightUpdateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val numberEntries = inputDimension * outputDimension
    val seriesAccumulator = DenseAccumulator(numberEntries)
    val batchAccumulator = DenseAccumulator(numberEntries)

    val stepProjections = Array(numberSteps) { indexStep ->

        val stepProjectionName = concatenateNames(stepName, indexStep.toString())

        if (useIdentityAtFirstStep && indexStep == 0) {

            createIdentityLayer(stepProjectionName)

        }
        else {

            createStepWeighting(stepProjectionName, numberWeightRows, numberWeightColumns, weights, seriesAccumulator, weightUpdateRule)
        }

    }

    val updateRule = optimizationStrategy?.invoke(inputDimension, outputDimension)

    val seriesProjection = SeriesWeighting(
        seriesName,
        stepProjections,
        weights,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

    return seriesProjection

}

fun createStepWeighting(
    name : String?,
    numberWeightRows: Int,
    numberWeightColumns: Int,
    weights : DoubleArray,
    weightAccumulator: DenseAccumulator,
    weightUpdateRule : UpdateRule? = null): ProjectionLayer {

    return ProjectionLayer(name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule)

}