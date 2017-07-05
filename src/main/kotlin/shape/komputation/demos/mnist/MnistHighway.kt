package shape.komputation.demos.mnist

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.functions.findMaxIndex
import shape.komputation.initialization.constantInitialization
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.denseLayer
import shape.komputation.layers.forward.highwayLayer
import shape.komputation.loss.logisticLoss
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
    val hiddenDimension = 100

    val gaussianInitialization = gaussianInitialization(random, 0.0,0.0001)
    val zeroInitialization = zeroInitialization()
    val constantInitialization = constantInitialization(-2.0)

    val optimizer = momentum(0.01, 0.1)

    val dimensionalityReductionLayer = denseLayer(
        inputDimension,
        hiddenDimension,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Tanh,
        optimizer
    )

    val outputLayer = denseLayer(
        hiddenDimension,
        MnistData.numberCategories,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val createHiddenLayer = { highwayLayer(hiddenDimension, gaussianInitialization, zeroInitialization, constantInitialization, ActivationFunction.Tanh, optimizer) }

    val network = Network(
        inputLayer(),
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
            .div(MnistData.numberTestExamples.toDouble())

        println(accuracy)

    }

    network.train(trainingInputs, trainingTargets, logisticLoss(), numberIterations, batchSize, afterEachIteration)

}