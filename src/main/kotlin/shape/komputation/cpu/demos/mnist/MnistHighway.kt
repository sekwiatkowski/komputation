package shape.komputation.cpu.demos.mnist

import shape.komputation.cpu.Network
import shape.komputation.cpu.functions.findMaxIndex
import shape.komputation.demos.mnist.MnistData
import shape.komputation.initialization.constantInitialization
import shape.komputation.initialization.gaussianInitialization
import shape.komputation.initialization.zeroInitialization
import shape.komputation.layers.entry.inputLayer
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.denseLayer
import shape.komputation.layers.forward.highwayLayer
import shape.komputation.loss.logisticLoss
import shape.komputation.optimization.historical.momentum
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
    val numberCategories = MnistData.numberCategories

    val gaussianInitialization = gaussianInitialization(random, 0.0f,0.0001f)
    val zeroInitialization = zeroInitialization()
    val constantInitialization = constantInitialization(-2.0f)

    val optimizer = momentum(0.01f, 0.1f)

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
        numberCategories,
        gaussianInitialization,
        gaussianInitialization,
        ActivationFunction.Softmax,
        optimizer
    )

    val createHiddenLayer = {

        highwayLayer(hiddenDimension, gaussianInitialization, zeroInitialization, constantInitialization, ActivationFunction.Tanh, optimizer)

    }

    val network = Network(
        inputLayer(inputDimension),
        dimensionalityReductionLayer,
        createHiddenLayer(),
        createHiddenLayer(),
        createHiddenLayer(),
        outputLayer
    )

    val afterEachIteration = { _ : Int, _ : Float ->

        val accuracy = network
            .test(
                testInputs,
                testTargets,
                batchSize,
                { prediction, target ->

                    findMaxIndex(prediction) == findMaxIndex(target)

                }
            )
            .count { correct -> correct }
            .toFloat()
            .div(MnistData.numberTestExamples.toFloat())

        println(accuracy)

    }

    network.train(trainingInputs, trainingTargets, logisticLoss(numberCategories), numberIterations, batchSize, afterEachIteration)

}