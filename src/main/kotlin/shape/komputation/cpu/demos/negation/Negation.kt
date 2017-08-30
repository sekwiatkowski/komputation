package shape.komputation.cpu.demos.negation

import shape.komputation.cpu.network.Network
import shape.komputation.loss.printLoss
import shape.komputation.demos.negation.NegationData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.dense.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 1
    val outputDimension = 1
    val batchSize = 2

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.01f)

    Network(
        batchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )
        .training(
            NegationData.inputs,
            NegationData.targets,
            10_000,
            squaredLoss(outputDimension),
            printLoss)
        .run()

}