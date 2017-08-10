package shape.komputation.cuda.demos.xor

import shape.komputation.cpu.printLoss
import shape.komputation.cuda.CudaNetwork
import shape.komputation.demos.xor.XorData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
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
        inputLayer,
        hiddenLayer,
        outputLayer
    )

    network.train(XorData.input, XorData.targets, squaredLoss(outputDimension), 10_000, batchSize, printLoss)

}