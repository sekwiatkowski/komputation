package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cpu.layers.CpuLayerState
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.initialization.InitializationStrategy
import com.komputation.initialization.initializeWeights
import com.komputation.layers.concatenateNames
import com.komputation.optimization.OptimizationInstruction

class SeriesWeighting internal constructor(
    private val name : String?,
    override val numberInputRows: Int,
    override val numberInputColumns: Int,
    override val numberOutputRows: Int,
    private val layers: Array<CpuForwardLayer>,
    private val weights: FloatArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule?) : CpuLayerState {
    private val numberWeightEntries = this.weights.size

    override val numberOutputColumns = this.numberInputColumns
    override var forwardResult = FloatArray(0)

    override var backwardResult = FloatArray(0)

    fun getWeights() =
        this.weights

    fun forwardStep(withinBatch : Int, step : Int, input: FloatArray, isTraining : Boolean): FloatArray {
        this.forwardResult = this.layers[step].forward(withinBatch, this.numberInputColumns, input, isTraining)

        return this.forwardResult
    }

    fun backwardStep(withinBatch : Int, step: Int, chain: FloatArray): FloatArray {
        val weightingLayer = this.layers[step]

        weightingLayer.backward(withinBatch, chain)

        this.backwardResult = weightingLayer.backwardResult

        return this.backwardResult
    }

    fun backwardSeries() {
        val seriesAccumulation = this.seriesAccumulator.getAccumulation()
        this.batchAccumulator.accumulate(seriesAccumulation)
        this.seriesAccumulator.reset()
    }

    fun optimize(batchSize : Int) {
        if (this.updateRule != null) {
            val accumulation = this.batchAccumulator.getAccumulation()
            updateDensely(this.weights, accumulation, this.numberWeightEntries, batchSize, this.updateRule)
        }

        this.batchAccumulator.reset()
    }
}

fun seriesWeighting(
    numberSteps : Int,
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationInstruction?) =

    seriesWeighting(
        null,
        numberSteps,
        numberInputRows,
        numberInputColumns,
        numberOutputRows,
        initializationStrategy,
        optimizationStrategy
    )

fun seriesWeighting(
    seriesName: String?,
    numberSteps : Int,
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction?) : SeriesWeighting {

    val numberWeightRows = numberOutputRows
    val numberWeightColumns = numberInputRows
    val weights = initializeWeights(initialization, numberWeightRows, numberWeightColumns, numberInputRows)

    val numberWeightEntries = numberWeightRows * numberWeightColumns
    val seriesAccumulator = DenseAccumulator(numberWeightEntries)

    val weightingLayers = Array<CpuForwardLayer>(numberSteps) { indexStep ->

        val stepName = concatenateNames(seriesName, indexStep.toString())

        CpuWeightingLayer(stepName, weights, numberInputRows, numberInputColumns, numberInputColumns, numberWeightRows, seriesAccumulator)
    }

    val batchAccumulator = DenseAccumulator(numberWeightEntries)

    val optimizationStrategy = optimization?.buildForCpu()
    val updateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val seriesWeighting = SeriesWeighting(
        seriesName,
        numberInputRows,
        numberInputColumns,
        numberOutputRows,
        weightingLayers,
        weights,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

    return seriesWeighting

}