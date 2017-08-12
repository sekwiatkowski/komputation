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
    val batchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    Network(
            batchSize,
            inputLayer(inputDimension),
            denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
        )
        .training(
            BinaryAndData.inputs,
            BinaryAndData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)
        .run()

}