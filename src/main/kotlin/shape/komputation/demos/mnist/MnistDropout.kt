package shape.komputation.demos.mnist

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.findMaxIndex
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.activation.ReluLayer
import shape.komputation.layers.forward.createDenseLayer
import shape.komputation.layers.forward.dropout.createDropoutLayer
import shape.komputation.layers.forward.projection.createProjectionLayer
import shape.komputation.loss.LogisticLoss
import shape.komputation.networks.Network
import shape.komputation.optimization.momentum
import java.io.File
import java.util.*

fun main(args: Array<String>) {

    if (args.size != 2) {

        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")

    }

    val random = Random(1)

    val numberIterations = 10
    val batchSize = 1

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()))
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()))

    val inputDimension = 784
    val hiddenDimension = 500

    val gaussianInitialization = createGaussianInitializer(random, 0.0,0.01)

    val optimizer = momentum(0.01, 0.1)

    val firstProjection = createProjectionLayer(
        inputDimension,
        hiddenDimension,
        gaussianInitialization,
        gaussianInitialization,
        optimizer
    )


    val outputLayer = createDenseLayer(
        hiddenDimension,
        MnistData.numberCategories,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val network = Network(
        InputLayer(),
        firstProjection,
        createDropoutLayer(hiddenDimension, random, 0.9, ReluLayer()),
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

    network.train(trainingInputs, trainingTargets, LogisticLoss(), numberIterations, batchSize, afterEachIteration)

}