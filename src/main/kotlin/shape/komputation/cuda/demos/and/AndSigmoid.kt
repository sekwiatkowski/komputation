package shape.komputation.cuda.demos.and

import shape.komputation.cuda.network.CudaNetwork
import shape.komputation.demos.and.BinaryAndData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.dense.denseLayer
import shape.komputation.loss.printLoss
import shape.komputation.loss.squaredLoss
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 1
    val maximumBatchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    val network = CudaNetwork(
        maximumBatchSize,
        inputLayer(inputDimension),
        denseLayer(inputDimension, outputDimension, initialization, initialization, ActivationFunction.Sigmoid, optimization)
    )

    val training = network.training(BinaryAndData.inputs, BinaryAndData.targets, 10_000, squaredLoss(outputDimension), printLoss)

    training.run()

    training.free()
    network.free()

}