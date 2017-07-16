package shape.komputation.cpu.demos.and

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.and.BinaryAndData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 1

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03)

    val network = Network(
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    network.train(BinaryAndData.inputs, BinaryAndData.targets, squaredLoss(outputDimension), 10_000, 1, printLoss)

}