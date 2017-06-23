package shape.komputation.layers.feedforward.recurrent

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.layers.ContinuationLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class SeriesProjection(
    private val name : String?,
    private val projections: Array<ContinuationLayer>,
    private val weights: DoubleArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule?) {

    private val optimize = updateRule != null

    fun forwardStep(step : Int, input: DoubleMatrix): DoubleMatrix {

        return projections[step].forward(input)

    }

    fun backwardStep(step: Int, chain: DoubleMatrix) : DoubleMatrix {

        return projections[step].backward(chain)

    }

    fun backwardSeries() {

        val seriesAccumulator = this.seriesAccumulator

        this.batchAccumulator.accumulate(seriesAccumulator.getAccumulation())

        seriesAccumulator.reset()

    }

    fun optimize() {

        val batchAccumulator = this.batchAccumulator

        if (this.optimize) {

            updateDensely(this.weights, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), this.updateRule!!)

        }

        batchAccumulator.reset()

    }

}

fun createSeriesProjection(
    name : String?,
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    inputRows: Int,
    outputRows: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationStrategy?) : Pair<SeriesProjection, Array<ContinuationLayer>> {

    val weights = initializeMatrix(initializationStrategy, outputRows, inputRows)

    val numberEntries = inputRows * outputRows
    val seriesAccumulator = DenseAccumulator(numberEntries)
    val batchAccumulator = DenseAccumulator(numberEntries)

    val stepProjections = createStepProjections(name, numberSteps, useIdentityAtFirstStep, weights, inputRows, outputRows, seriesAccumulator)

    val updateRule = if(optimizationStrategy != null) optimizationStrategy(inputRows, outputRows) else null

    val seriesProjection = SeriesProjection(
        name,
        stepProjections,
        weights,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

    return seriesProjection to stepProjections

}