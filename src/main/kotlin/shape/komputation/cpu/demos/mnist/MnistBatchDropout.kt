package shape.komputation.cpu.demos.mnist

import shape.komputation.cpu.Network
import shape.komputation.demos.mnist.MnistData
import shape.komputation.initialization.heInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.layers.forward.dropout.dropoutLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.historical.momentum
import java.io.File
import java.util.*

// The data set for this demo can be found here: https://pjreddie.com/projects/mnist-in-csv/
fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")

    }

    val random = Random(1)

    val numberIterations = 50
    val batchSize = 64

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()))
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()))

    val inputDimension = 784
    val hiddenDimension = 100
    val numberCategories = MnistData.numberCategories

    val initialization = heInitialization(random)
    val optimizer = momentum(0.01f, 0.9f)
    val keepProbability = 0.85f

    val hiddenLayer = denseLayer(
        inputDimension,
        hiddenDimension,
        initialization,
        initialization,
        ActivationFunction.ReLU,
        optimizer
    )

    val outputLayer = denseLayer(
        hiddenDimension,
        numberCategories,
        initialization,
        initialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val network = Network(
        inputLayer(inputDimension),
        hiddenLayer,
        dropoutLayer(random, keepProbability, hiddenDimension),
        outputLayer
    )

    val afterEachIteration = { _ : Int, _ : Float ->

        val accuracy = network
            .test(
                testInputs,
                testTargets,
                batchSize,
                numberCategories)

        println(accuracy)

    }

    network.train(trainingInputs, trainingTargets, logisticLoss(numberCategories), numberIterations, batchSize, afterEachIteration)

}