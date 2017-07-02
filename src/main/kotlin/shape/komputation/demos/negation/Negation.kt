package shape.komputation.demos.negation

import shape.komputation.initialization.createUniformInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.activation.SigmoidLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleScalar
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

object NegationData {

    val inputs = arrayOf<Matrix>(
        doubleScalar(0.0),
        doubleScalar(1.0)
    )

    val targets = arrayOf(
        doubleScalar(1.0),
        doubleScalar(0.0)
    )

}

fun main(args: Array<String>) {

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val optimizationStrategy = stochasticGradientDescent(0.01)

    val projectionLayer = createProjectionLayer(1, 1, initialize, initialize, optimizationStrategy)
    val sigmoidLayer = SigmoidLayer()

    val network = Network(
        InputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    network.train(NegationData.inputs, NegationData.targets, SquaredLoss(), 10_000, 1, printLoss)

}