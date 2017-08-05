package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.layers.CpuForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.identityLayer
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.OptimizationInstruction

class SeriesBias internal constructor(
    private val name : String?,
    private val biases: Array<CpuBiasLayer>,
    private val bias: FloatArray,
    private val seriesAccumulator: DenseAccumulator,
    private val batchAccumulator: DenseAccumulator,
    private val updateRule: UpdateRule? = null) {

    private val numberBiasEntries = this.bias.size

    fun forwardStep(withinBatch : Int, step : Int, input : FloatMatrix, isTraining : Boolean) =

        this.biases[step].forward(withinBatch, input, isTraining)


    fun backwardStep(withinBatch: Int, step : Int, chain: FloatMatrix)  =

        this.biases[step].backward(withinBatch, chain)

    fun backwardSeries() {

        this.batchAccumulator.accumulate(this.seriesAccumulator.getAccumulation())
        this.seriesAccumulator.reset()

    }

    fun optimize(scalingFactor : Float) {

        if (this.updateRule != null) {

            updateDensely(this.bias, this.batchAccumulator.getAccumulation(), this.numberBiasEntries, scalingFactor, this.updateRule)

        }

        this.batchAccumulator.reset()

    }

}

fun seriesBias(
    numberSteps: Int,
    numberInputRows: Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction?) =

    seriesBias(null, null, numberSteps, numberInputRows, initialization, optimization)

fun seriesBias(
    seriesName: String?,
    stepNamePrefix: String?,
    numberSteps: Int,
    numberInputRows: Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction?) : SeriesBias {

    val bias = initializeColumnVector(initialization, numberInputRows)

    val seriesAccumulator = DenseAccumulator(numberInputRows)

    val biasLayers = Array(numberSteps) { indexStep ->

        val stepName = concatenateNames(stepNamePrefix, indexStep.toString())

        CpuBiasLayer(stepName, numberInputRows, 1, bias, seriesAccumulator)

    }

    val batchAccumulator = DenseAccumulator(numberInputRows)

    val optimizationStrategy = optimization?.buildForCpu()
    val updateRule = optimizationStrategy?.invoke(numberInputRows, 1)

    return SeriesBias(
        seriesName,
        biasLayers,
        bias,
        seriesAccumulator,
        batchAccumulator,
        updateRule)

}