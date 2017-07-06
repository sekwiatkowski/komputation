package shape.komputation.demos.xor

import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
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

    val inputLayer = inputLayer()

    val initialize = heInitialization(random)
    val optimizationStrategy = stochasticGradientDescent(0.1)

    val hiddenPreactivationLayer = projectionLayer(2, 2, initialize, initialize, optimizationStrategy)
    val hiddenLayer = sigmoidLayer()

    val outputPreactivationLayer = projectionLayer(2, 1, initialize, initialize, optimizationStrategy)
    val outputLayer = sigmoidLayer()

    val network = Network(
        inputLayer,
        hiddenPreactivationLayer,
        hiddenLayer,
        outputPreactivationLayer,
        outputLayer
    )

    network.train(XorData.input, XorData.targets, squaredLoss(), 30_000, 1, printLoss)

}