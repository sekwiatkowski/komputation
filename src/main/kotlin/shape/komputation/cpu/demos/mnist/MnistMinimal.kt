package shape.komputation.cpu.demos.mnist

import shape.komputation.cpu.Network
import shape.komputation.demos.mnist.MnistData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.dense.denseLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.historical.momentum
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")

    }

    val random = Random(1)

    val numberIterations = 30
    val batchSize = 64

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()))
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()))

    val inputDimension = 784
    val numberCategories = MnistData.numberCategories

    val initialization = gaussianInitialization(random, 0.0f, 0.1f)
    val optimizer = momentum(0.005f, 0.1f)

    val outputLayer = denseLayer(
        inputDimension,
        numberCategories,
        initialization,
        initialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val network = Network(
        batchSize,
        inputLayer(inputDimension),
        outputLayer
    )

    val test = network
        .test(
            testInputs,
            testTargets,
            batchSize,
            numberCategories)

    network.training(
        trainingInputs,
        trainingTargets,
        numberIterations,
        logisticLoss(numberCategories)) { _ : Int, _ : Float ->

            println(test.run())

        }
        .run()

}