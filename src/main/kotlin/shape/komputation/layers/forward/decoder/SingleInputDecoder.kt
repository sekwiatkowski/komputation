package shape.komputation.layers.forward.decoder

import shape.komputation.cpu.forward.activation.activationLayer
import shape.komputation.cpu.forward.decoder.CpuSingleInputDecoder
import shape.komputation.cpu.forward.projection.seriesBias
import shape.komputation.cpu.forward.projection.seriesWeighting
import shape.komputation.cpu.forward.units.RecurrentUnit
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.optimization.OptimizationStrategy


class SingleInputDecoder(
    private val name : String?,
    private val numberSteps: Int,
    private val hiddenDimension : Int,
    private val outputDimension: Int,
    private val unit : RecurrentUnit,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimizationStrategy: OptimizationStrategy?): CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuSingleInputDecoder {

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

        val decoder = CpuSingleInputDecoder(
            this.name,
            this.numberSteps,
            this.outputDimension,
            this.unit,
            weighting,
            bias,
            activations)

        return decoder

    }

}

fun singleInputDecoder(
    numberSteps: Int,
    hiddenDimension : Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    singleInputDecoder(
        null,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)


fun singleInputDecoder(
    name : String?,
    numberSteps: Int,
    hiddenDimension : Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    SingleInputDecoder(
        name,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)