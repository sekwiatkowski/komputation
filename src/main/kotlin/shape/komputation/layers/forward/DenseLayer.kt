package shape.komputation.layers.forward

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.layers.forward.activation.activationLayer
import shape.komputation.layers.forward.projection.CpuProjectionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class CpuDenseLayer internal constructor(
    name : String?,
    private val projection : CpuProjectionLayer,
    private val activation: ActivationLayer) : BaseForwardLayer(name), Optimizable {

    override fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        val projected = this.projection.forward(input, isTraining)

        val activated = this.activation.forward(projected, isTraining)

        return activated
    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val diffChainWrtActivation = this.activation.backward(chain)

        val diffActivationWrtProjection = this.projection.backward(diffChainWrtActivation)

        return diffActivationWrtProjection

    }

    override fun optimize(scalingFactor : Double) {

        this.projection.optimize(scalingFactor)

    }

}

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
        val activation = activationLayer(activationName, activationFunction)

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