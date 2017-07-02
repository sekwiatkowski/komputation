package shape.komputation.demos.mnist

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.findMaxIndex
import shape.komputation.initialization.createConstantInitializer
import shape.komputation.initialization.createGaussianInitializer
import shape.komputation.initialization.createZeroInitializer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.forward.createDenseLayer
import shape.komputation.layers.forward.createHighwayLayer
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

    val gaussianInitialization = createGaussianInitializer(random, 0.0,0.0001)
    val zeroInitialization = createZeroInitializer()
    val constantInitialization = createConstantInitializer(-2.0)

    val optimizer = momentum(0.01, 0.1)

    val hiddenDimension = 100

    val dimensionalityReductionLayer = createDenseLayer(
        784,
        hiddenDimension,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Tanh,
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

    val createHiddenLayer = { createHighwayLayer(hiddenDimension, gaussianInitialization, zeroInitialization, constantInitialization, ActivationFunction.Tanh, optimizer) }

    val network = Network(
        InputLayer(),
        dimensionalityReductionLayer,
        createHiddenLayer(),
        createHiddenLayer(),
        createHiddenLayer(),
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
            .div(10_000.0)

        println(accuracy)

    }
    network.train(trainingInputs, trainingTargets, LogisticLoss(), numberIterations, batchSize, afterEachIteration)

}