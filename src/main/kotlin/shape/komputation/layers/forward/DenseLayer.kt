package shape.komputation.layers.forward

import shape.komputation.cpu.forward.CpuDenseLayer
import shape.komputation.cpu.forward.activation.activationLayer
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.optimization.OptimizationStrategy

class DenseLayer(
    private val name : String?,
    private val inputDimension : Int,
    private val outputDimension : Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimizationStrategy: OptimizationStrategy?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuDenseLayer {

        val projectionName = concatenateNames(name, "projection")
        val projection = projectionLayer(projectionName, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy).buildForCpu()

        val activationName = concatenateNames(name, "activation")
        val activation = activationLayer(activationName, activationFunction).buildForCpu()

        return CpuDenseLayer(name, projection, activation)

    }


}

fun denseLayer(
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

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
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    DenseLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy)