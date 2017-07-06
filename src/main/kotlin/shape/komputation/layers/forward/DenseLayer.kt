package shape.komputation.layers.forward

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.layers.forward.activation.activationLayer
import shape.komputation.layers.forward.projection.ProjectionLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class DenseLayer internal constructor(
    name : String?,
    private val projection : ProjectionLayer,
    private val activation: ActivationLayer) : ForwardLayer(name), Optimizable {

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
    optimizationStrategy: OptimizationStrategy?): DenseLayer {

    val projectionName = concatenateNames(name, "projection")
    val projection = projectionLayer(projectionName, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

    val activationName = concatenateNames(name, "activation")
    val activation = activationLayer(activationName, activationFunction)

    return DenseLayer(name, projection, activation)

}