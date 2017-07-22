package shape.komputation.cpu.demos.xor

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.xor.XorData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val hiddenDimension = 2
    val outputDimension = 1

    val random = Random(1)

    val inputLayer = inputLayer(inputDimension)

    val initialization = heInitialization(random)
    val optimization = stochasticGradientDescent(0.1f)

    val hiddenLayer = denseLayer(inputDimension, hiddenDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    val outputLayer = denseLayer(hiddenDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)

    val network = Network(
        inputLayer,
        hiddenLayer,
        outputLayer
    )

    network.train(XorData.input, XorData.targets, squaredLoss(outputDimension), 30_000, 1, printLoss)

}