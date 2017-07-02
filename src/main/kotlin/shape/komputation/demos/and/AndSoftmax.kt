package shape.komputation.demos.and

import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.activation.SoftmaxLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.loss.LogisticLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(
        doubleColumnVector(0.0, 0.0),
        doubleColumnVector(0.0, 1.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 1.0)
    )

    val targets = arrayOf(
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(1.0, 0.0),
        doubleColumnVector(0.0, 1.0)
    )

    val random = Random(1)
    val initialize = createGaussianInitializer(random)

    val optimizer = stochasticGradientDescent(0.03)

    val projectionLayer = createProjectionLayer(2, 2, initialize, initialize, optimizer)
    val softmaxLayer = SoftmaxLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        softmaxLayer
    )

    network.train(input, targets, LogisticLoss(), 10_000, 1, printLoss)

}