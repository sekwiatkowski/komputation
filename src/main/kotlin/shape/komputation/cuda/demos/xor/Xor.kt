package shape.komputation.cuda.demos.xor

import shape.komputation.loss.printLoss
import shape.komputation.cuda.network.CudaNetwork
import shape.komputation.demos.xor.XorData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.dense.denseLayer
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.historical.nesterov
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val hiddenDimension = 2
    val outputDimension = 1
    val batchSize = 4

    val random = Random(1)

    val inputLayer = inputLayer(inputDimension)

    val initialization = heInitialization(random)
    val optimization = nesterov(0.1f, 0.9f)

    val hiddenLayer = denseLayer(inputDimension, hiddenDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    val outputLayer = denseLayer(hiddenDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)

    val network = CudaNetwork(
        batchSize,
        inputLayer,
        hiddenLayer,
        outputLayer
    )

    val training = network.training(XorData.input, XorData.targets, 10_000, squaredLoss(outputDimension), printLoss)

    training.run()

    training.free()
    network.free()

}