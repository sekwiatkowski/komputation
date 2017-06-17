package shape.komputation.demos

import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.feedforward.activation.SoftmaxLayer
import shape.komputation.layers.feedforward.projection.createProjectionLayer
import shape.komputation.loss.LogisticLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleRowVector
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(
        doubleRowVector(0.0, 0.0),
        doubleRowVector(0.0, 1.0),
        doubleRowVector(1.0, 0.0),
        doubleRowVector(1.0, 1.0)
    )

    val targets = arrayOf(
        doubleRowVector(1.0, 0.0),
        doubleRowVector(1.0, 0.0),
        doubleRowVector(1.0, 0.0),
        doubleRowVector(0.0, 1.0)
    )

    val random = Random(1)
    val initialize = createGaussianInitializer(random)

    val optimizer = stochasticGradientDescent(0.03)

    val projectionLayer = createProjectionLayer(2, 2, true, initialize, optimizer)
    val softmaxLayer = SoftmaxLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        softmaxLayer
    )

    network.train(input, targets, LogisticLoss(), 10_000, 1, printLoss)

}