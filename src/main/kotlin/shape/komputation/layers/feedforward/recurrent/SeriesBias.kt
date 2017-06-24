package shape.komputation.layers.feedforward.recurrent

import shape.komputation.functions.add
import shape.komputation.functions.backwardProjectionWrtBias
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeRowVector
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateDensely

class SeriesBias(
    private val name : String?,
    private val bias: DoubleArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule? = null) {

    private val optimizes = updateRule != null

    fun forwardStep(input : DoubleArray) =

        add(input, bias)

    fun backwardStep(chain: DoubleMatrix) {

        val backwardWrtBias = backwardProjectionWrtBias(this.bias.size, chain.entries, chain.numberRows, chain.numberColumns)

        this.seriesAccumulator.accumulate(backwardWrtBias)

    }

    fun backwardSeries() {

        val seriesAccumulator = this.seriesAccumulator

        this.batchAccumulator.accumulate(seriesAccumulator.getAccumulation())

        seriesAccumulator.reset()

    }

    fun optimize() {

        val batchAccumulator = this.batchAccumulator

        if (this.optimizes) {

            updateDensely(this.bias, batchAccumulator.getAccumulation(), batchAccumulator.getCount(), updateRule!!)

        }

        batchAccumulator.reset()

    }

}

fun createSeriesBias(
    name : String?,
    dimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationStrategy?) : SeriesBias {

    val bias = initializeRowVector(initializationStrategy, dimension)

    val seriesAccumulator = DenseAccumulator(dimension)
    val batchAccumulator = DenseAccumulator(dimension)

    val updateRule = optimizationStrategy?.invoke(dimension, 1)

    return SeriesBias(name, bias, seriesAccumulator, batchAccumulator, updateRule)

}