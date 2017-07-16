package shape.komputation.cpu.demos.xor

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.xor.XorData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

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