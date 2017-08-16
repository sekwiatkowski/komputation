package shape.komputation.cuda.demos.negation

import shape.komputation.cpu.printLoss
import shape.komputation.cuda.CudaNetwork
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

    val network = CudaNetwork(
        batchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    val training = network.training(NegationData.inputs, NegationData.targets, 10_000, squaredLoss(outputDimension), printLoss)
    training.run()

    training.free()
    network.free()

}