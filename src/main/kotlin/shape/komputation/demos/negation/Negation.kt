package shape.komputation.demos.negation

import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
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
    val initialize = heInitialization(random)

    val optimizationStrategy = stochasticGradientDescent(0.01)

    val projectionLayer = projectionLayer(1, 1, initialize, initialize, optimizationStrategy)
    val sigmoidLayer = sigmoidLayer()

    val network = Network(
        inputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    network.train(NegationData.inputs, NegationData.targets, squaredLoss(), 10_000, 1, printLoss)

}