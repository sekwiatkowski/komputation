package shape.komputation.cpu.demos.mnist

import shape.komputation.cpu.Network
import shape.komputation.cpu.functions.findMaxIndex
import shape.komputation.demos.mnist.MnistData
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.denseLayer
import shape.komputation.layers.forward.dropout.dropoutLayer
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.loss.logisticLoss
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
    val numberCategories = MnistData.numberCategories

    val initialization = gaussianInitialization(random, 0.0f, 0.1f)

    val optimizer = nesterov(0.005f, 0.1f)

    val firstProjection = projectionLayer(
        inputDimension,
        hiddenDimension,
        initialization,
        initialization,
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
        firstProjection,
        dropoutLayer(random, hiddenDimension, 0.85f),
        reluLayer(hiddenDimension),
        outputLayer
    )

    val afterEachIteration = { _ : Int, _ : Float ->

        val accuracy = network
            .test(
                testInputs,
                testTargets,
                { prediction, target ->

                    findMaxIndex(prediction.entries) == findMaxIndex(target.entries)

                }
            )
            .count { correct -> correct }
            .toFloat()
            .div(MnistData.numberTestExamples.toFloat())

        println(accuracy)

    }

    network.train(trainingInputs, trainingTargets, logisticLoss(numberCategories), numberIterations, batchSize, afterEachIteration)

}