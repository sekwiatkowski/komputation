package shape.komputation.layers.recurrent

import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeMatrix
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.feedforward.projection.createIdentityProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class SeriesProjection(
    private val projections: Array<FeedForwardLayer>,
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

        val stateWeightSeriesAccumulator = this.seriesAccumulator

        this.batchAccumulator.accumulate(stateWeightSeriesAccumulator.getAccumulation())

        stateWeightSeriesAccumulator.reset()

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
    numberWeightRows: Int,
    numberWeightColumns: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationStrategy?) : SeriesProjection {

    val weights = initializeMatrix(initializationStrategy, numberWeightRows, numberWeightColumns)
    val numberEntries = numberWeightRows * numberWeightColumns
    val seriesAccumulator = DenseAccumulator(numberEntries)
    val weightAccumulator = DenseAccumulator(numberEntries)
    val updateRule = if(optimizationStrategy != null) optimizationStrategy(numberWeightRows, numberWeightColumns) else null

    val stateStepProjections = Array(numberSteps) { index ->

        val stateProjectionLayerName = if (name == null) null else "$name-$index"

        if (useIdentityAtFirstStep && index == 0) {

            createIdentityProjectionLayer(stateProjectionLayerName)

        }
        else {

            createStepProjection(stateProjectionLayerName, weights, numberWeightRows, numberWeightColumns, seriesAccumulator)
        }

    }

    val seriesProjection = SeriesProjection(
        stateStepProjections,
        weights,
        seriesAccumulator,
        weightAccumulator,
        updateRule

    )

    return seriesProjection

}