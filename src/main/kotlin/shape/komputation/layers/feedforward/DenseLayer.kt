package shape.komputation.layers.feedforward

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.feedforward.activation.ActivationLayer
import shape.komputation.layers.feedforward.activation.createActivationLayer
import shape.komputation.layers.feedforward.projection.ProjectionLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class DenseLayer(
    name : String?,
    private val projection : ProjectionLayer,
    private val activation: ActivationLayer) : ContinuationLayer(name), Optimizable {

    override fun forward(input: DoubleMatrix): DoubleMatrix {

        val projected = this.projection.forward(input)

        val activated = this.activation.forward(projected)

        return activated
    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        val diffChainWrtActivation = this.activation.backward(chain)

        val diffActivationWrtProjection = this.projection.backward(diffChainWrtActivation)

        return diffActivationWrtProjection

    }

    override fun optimize() {

        this.projection.optimize()

    }

}

fun createDenseLayer(
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    createDenseLayer(
        null,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy
    )

fun createDenseLayer(
    name : String?,
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?): DenseLayer {

    val projectionName = concatenateNames(name, "projection")
    val projection = createProjectionLayer(projectionName, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

    val activationName = concatenateNames(name, "activation")
    val activation = createActivationLayer(activationName, activationFunction)

    return DenseLayer(name, projection, activation)


}