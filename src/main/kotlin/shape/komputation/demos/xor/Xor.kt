package shape.komputation.demos.xor

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.matrix.doubleScalar
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

    val inputDimension = 2
    val hiddenDimension = 2
    val outputDimension = 1

    val random = Random(1)

    val inputLayer = inputLayer(inputDimension)

    val initialize = heInitialization(random)
    val optimizationStrategy = stochasticGradientDescent(0.1)

    val hiddenPreactivationLayer = projectionLayer(inputDimension, hiddenDimension, initialize, initialize, optimizationStrategy)
    val hiddenActivationLayer = sigmoidLayer(hiddenDimension)

    val outputPreactivationLayer = projectionLayer(hiddenDimension, outputDimension, initialize, initialize, optimizationStrategy)
    val outputActivationLayer = sigmoidLayer(outputDimension)

    val network = Network(
        inputLayer,
        hiddenPreactivationLayer,
        hiddenActivationLayer,
        outputPreactivationLayer,
        outputActivationLayer
    )

    network.train(XorData.input, XorData.targets, squaredLoss(outputDimension), 30_000, 1, printLoss)

}