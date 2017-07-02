package shape.komputation.layers.forward

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationLayer
import shape.komputation.layers.forward.activation.createActivationLayer
import shape.komputation.layers.forward.projection.ProjectionLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class DenseLayer(
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