package shape.komputation.demos.mnist

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.findMaxIndex
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.denseLayer
import shape.komputation.layers.forward.dropout.dropoutLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.networks.Network
import shape.komputation.optimization.adaptive.adam
import shape.komputation.optimization.historical.nesterov
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")

    }

    val random = Random(1)

    val numberIterations = 30
    val batchSize = 1

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()))
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()))

    val inputDimension = 784
    val hiddenDimension = 100

    val initialization = gaussianInitialization(random, 0.0, 0.1)

    val optimizer = nesterov(0.005, 0.1)

    val firstProjection = projectionLayer(
        inputDimension,
        hiddenDimension,
        initialization,
        initialization,
        optimizer
    )

    val outputLayer = denseLayer(
        hiddenDimension,
        MnistData.numberCategories,
        initialization,
        initialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val network = Network(
        inputLayer(),
        firstProjection,
        dropoutLayer(hiddenDimension, random, 0.85, reluLayer()),
        outputLayer
    )

    val afterEachIteration = { _ : Int, _ : Double ->

        val accuracy = network
            .test(
                testInputs,
                testTargets,
                { prediction, target ->

                    findMaxIndex(prediction.entries) == findMaxIndex(target.entries)

                }
            )
            .count { correct -> correct }
            .div(MnistData.numberTestExamples.toDouble())

        println(accuracy)

    }

    network.train(trainingInputs, trainingTargets, logisticLoss(), numberIterations, batchSize, afterEachIteration)

}