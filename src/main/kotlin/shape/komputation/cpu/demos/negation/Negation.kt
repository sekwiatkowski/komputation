package shape.komputation.cpu.demos.negation

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.negation.NegationData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 1
    val outputDimension = 1

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.01f)

    val network = Network(
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    network.train(NegationData.inputs, NegationData.targets, squaredLoss(outputDimension), 10_000, 2, printLoss)

}