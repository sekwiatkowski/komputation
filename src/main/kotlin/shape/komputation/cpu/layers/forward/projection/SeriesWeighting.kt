package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.CpuForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.identityLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.OptimizationInstruction

class SeriesWeighting internal constructor(
    private val name : String?,
    private val weightingLayers: Array<CpuForwardLayer>,
    private val weights: FloatArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule?) {

    private val numberWeightEntries = this.weights.size

    fun forwardStep(withinBatch : Int, step : Int, input: FloatMatrix, isTraining : Boolean) =

        this.weightingLayers[step].forward(withinBatch, input, isTraining)

    fun backwardStep(withinBatch : Int, step: Int, chain: FloatMatrix) =

        this.weightingLayers[step].backward(withinBatch, chain)

    fun backwardSeries() {

        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()

    }

    fun optimize(scalingFactor : Float) {

        if (this.updateRule != null) {

            updateDensely(this.weights, this.batchAccumulator.getAccumulation(), this.numberWeightEntries, scalingFactor, this.updateRule)

        }

        this.batchAccumulator.reset()

    }

}

fun seriesWeighting(
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy: OptimizationInstruction?) =

    seriesWeighting(
        null,
        null,
        numberSteps,
        useIdentityAtFirstStep,
        numberInputRows,
        numberInputColumns,
        numberOutputRows,
        initializationStrategy,
        optimizationStrategy
    )

fun seriesWeighting(
    seriesName: String?,
    stepNamePrefix: String?,
    numberSteps : Int,
    useIdentityAtFirstStep : Boolean,
    numberInputRows: Int,
    numberInputColumns: Int,
    numberOutputRows: Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction?) : SeriesWeighting {

    val numberWeightRows = numberOutputRows
    val numberWeightColumns = numberInputRows
    val weights = initializeWeights(initialization, numberWeightRows, numberWeightColumns, numberInputRows * numberInputColumns)

    val numberWeightEntries = numberWeightRows * numberWeightColumns
    val seriesAccumulator = DenseAccumulator(numberWeightEntries)

    val weightingLayers = Array<CpuForwardLayer>(numberSteps) { indexStep ->

        val stepName = concatenateNames(stepNamePrefix, indexStep.toString())

        if (useIdentityAtFirstStep && indexStep == 0) {

            identityLayer(stepName).buildForCpu()

        }
        else {

            CpuWeightingLayer(stepName, weights, numberInputRows, numberInputColumns, numberWeightRows, seriesAccumulator)

        }

    }

    val batchAccumulator = DenseAccumulator(numberWeightEntries)

    val optimizationStrategy = optimization?.buildForCpu()
    val updateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val seriesWeighting = SeriesWeighting(
        seriesName,
        weightingLayers,
        weights,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

    return seriesWeighting

}