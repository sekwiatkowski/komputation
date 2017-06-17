package shape.komputation.demos

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.activation.SigmoidLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleRowVector
import shape.komputation.matrix.doubleScalar
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(
        doubleRowVector(0.0, 0.0),
        doubleRowVector(1.0, 0.0),
        doubleRowVector(0.0, 1.0),
        doubleRowVector(1.0, 1.0))

    val targets = arrayOf(
        doubleScalar(0.0),
        doubleScalar(1.0),
        doubleScalar(1.0),
        doubleScalar(0.0)
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val inputLayer = InputLayer()

    val updateRule = stochasticGradientDescent(0.1)

    val hiddenPreactivationLayer = createProjectionLayer(2, 2, true, initialize, updateRule)
    val hiddenLayer = SigmoidLayer()

    val outputPreactivationLayer = createProjectionLayer(2, 1, true, initialize, updateRule)
    val outputLayer = SigmoidLayer()

    val network = Network(
        inputLayer,
        hiddenPreactivationLayer,
        hiddenLayer,
        outputPreactivationLayer,
        outputLayer
    )

    network.train(input, targets, SquaredLoss(), 30_000, 1, printLoss)

}