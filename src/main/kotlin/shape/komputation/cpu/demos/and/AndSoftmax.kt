package shape.komputation.cpu.demos.and

import shape.komputation.cpu.Network
import shape.komputation.cpu.printLoss
import shape.komputation.demos.and.OneHotAndData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 2

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    val network = Network(
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Softmax, optimization)
    )

    network.train(OneHotAndData.input, OneHotAndData.targets, logisticLoss(2), 10_000, 1, printLoss)

}