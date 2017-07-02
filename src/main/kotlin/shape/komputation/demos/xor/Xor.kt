package shape.komputation.demos.xor

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.activation.SigmoidLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleScalar
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

object XorData {

    val input = arrayOf<Matrix>(
        doubleColumnVector(0.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 1.0))

    val targets = arrayOf(
        doubleScalar(0.0),
        doubleScalar(1.0),
        doubleScalar(1.0),
        doubleScalar(0.0)
    )

}

fun main(args: Array<String>) {

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val inputLayer = InputLayer()

    val optimizationStrategy = stochasticGradientDescent(0.1)

    val hiddenPreactivationLayer = createProjectionLayer(2, 2, initialize, initialize, optimizationStrategy)
    val hiddenLayer = SigmoidLayer()

    val outputPreactivationLayer = createProjectionLayer(2, 1, initialize, initialize, optimizationStrategy)
    val outputLayer = SigmoidLayer()

    val network = Network(
        inputLayer,
        hiddenPreactivationLayer,
        hiddenLayer,
        outputPreactivationLayer,
        outputLayer
    )

    network.train(XorData.input, XorData.targets, SquaredLoss(), 30_000, 1, printLoss)

}