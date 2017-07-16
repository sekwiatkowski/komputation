package shape.komputation.layers.forward.decoder

import shape.komputation.cpu.layers.forward.activation.activationLayer
import shape.komputation.cpu.layers.forward.decoder.CpuSingleInputDecoder
import shape.komputation.cpu.layers.forward.projection.seriesBias
import shape.komputation.cpu.layers.forward.projection.seriesWeighting
import shape.komputation.cpu.layers.forward.units.RecurrentUnit
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.optimization.OptimizationInstruction


class SingleInputDecoder(
    private val name : String?,
    private val numberSteps: Int,
    private val hiddenDimension : Int,
    private val outputDimension: Int,
    private val unit : RecurrentUnit,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?): CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuSingleInputDecoder {

        val weightingSeriesName = concatenateNames(this.name, "weighting")
        val weightingStepName = concatenateNames(this.name, "weighting-step")
        val weighting = seriesWeighting(weightingSeriesName, weightingStepName, this.numberSteps, false, this.hiddenDimension, this.outputDimension, this.weightInitialization, optimization)

        val bias =

            if (this.biasInitialization != null) {

                val biasSeriesName = concatenateNames(this.name, "bias")
                seriesBias(biasSeriesName, this.outputDimension, biasInitialization, this.optimization)

            }
            else {

                null

            }

        val activationName = concatenateNames(this.name, "activation")
        val activations = Array(this.numberSteps) { index ->

            activationLayer(concatenateNames(activationName, index.toString()), this.activationFunction, this.outputDimension).buildForCpu()

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
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    singleInputDecoder(
        null,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)


fun singleInputDecoder(
    name : String?,
    numberSteps: Int,
    hiddenDimension : Int,
    outputDimension: Int,
    unit : RecurrentUnit,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: ActivationFunction,
    optimization: OptimizationInstruction?) =

    SingleInputDecoder(
        name,
        numberSteps,
        hiddenDimension,
        outputDimension,
        unit,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)