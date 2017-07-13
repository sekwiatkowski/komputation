package shape.komputation.layers.forward.decoder

import shape.komputation.cpu.forward.activation.activationLayer
import shape.komputation.cpu.forward.decoder.CpuMultiInputDecoder
import shape.komputation.cpu.forward.projection.seriesBias
import shape.komputation.cpu.forward.projection.seriesWeighting
import shape.komputation.cpu.forward.units.RecurrentUnit
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.optimization.OptimizationStrategy

class MultiInputDecoder(
    private val name : String?,
    private val numberSteps: Int,
    private val inputDimension: Int,
    private val hiddenDimension: Int,
    private val outputDimension: Int,
    private val unit : RecurrentUnit,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimizationStrategy: OptimizationStrategy?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuMultiInputDecoder {

        val weightingSeriesName = concatenateNames(this.name, "weighting")
        val weightingStepName = concatenateNames(this.name, "weighting-step")
        val weighting = seriesWeighting(weightingSeriesName, weightingStepName, this.numberSteps, false, this.hiddenDimension, this.outputDimension, this.weightInitializationStrategy, this.optimizationStrategy)

        val bias =
            if (this.biasInitializationStrategy != null) {

                val biasSeriesName = concatenateNames(this.name, "bias")
                seriesBias(biasSeriesName, this.outputDimension, biasInitializationStrategy, this.optimizationStrategy)

            }
            else {

                null

            }

        val activationName = concatenateNames(this.name, "activation")
        val activations = Array(this.numberSteps) { index ->

            activationLayer(concatenateNames(activationName, index.toString()), this.activationFunction).buildForCpu()

        }

        return CpuMultiInputDecoder(
            this.name,
            this.numberSteps,
            this.inputDimension,
            this.hiddenDimension,
            this.outputDimension,
            unit,
            weighting,
            bias,
            activations)

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
    optimizationStrategy: OptimizationStrategy?) =

    MultiInputDecoder(
        name,
        numberSteps,
        inputDimension,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)