package shape.komputation.layers.forward

import shape.komputation.cpu.functions.activation.ActivationFunction
import shape.komputation.cpu.layers.forward.CpuDenseLayer
import shape.komputation.cpu.layers.forward.activation.activationLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.optimization.OptimizationInstruction

class DenseLayer(
    private val name : String?,
    private val inputDimension : Int,
    private val outputDimension : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuDenseLayer {

        val projectionName = concatenateNames(this.name, "projection")
        val projection = projectionLayer(projectionName, this.inputDimension, this.outputDimension, this.weightInitialization, this.biasInitialization, this.optimization).buildForCpu()

        val activationName = concatenateNames(this.name, "activation")
        val activation = activationLayer(activationName, this.activationFunction, this.outputDimension).buildForCpu()

        return CpuDenseLayer(this.name, projection, activation)

    }


}

fun denseLayer(
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction?) =

    denseLayer(
        null,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy
    )

fun denseLayer(
    name : String?,
    inputDimension : Int,
    outputDimension : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimization: OptimizationInstruction?) =

    DenseLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitialization,
        biasInitialization,
        activationFunction,
        optimization)