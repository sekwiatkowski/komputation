package shape.komputation.demos.negation

import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.cublasProjectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.networks.Network
import shape.komputation.networks.printLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val initialize = heInitialization(random)

    val optimizationStrategy = stochasticGradientDescent(0.01)

    val projectionLayer = cublasProjectionLayer(1, 1, initialize, initialize, optimizationStrategy)
    val sigmoidLayer = sigmoidLayer()

    val network = Network(
        inputLayer(),
        projectionLayer,
        sigmoidLayer
    )

    network.train(NegationData.inputs, NegationData.targets, squaredLoss(), 10_000, 1, printLoss)

}