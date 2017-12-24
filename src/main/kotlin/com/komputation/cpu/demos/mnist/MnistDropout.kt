package com.komputation.cpu.demos.mnist

import com.komputation.cpu.network.Network
import com.komputation.demos.mnist.MnistData
import com.komputation.initialization.gaussianInitialization
import com.komputation.instructions.entry.input
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.continuation.dropout.dropout
import com.komputation.instructions.loss.crossEntropyLoss
import com.komputation.optimization.historical.momentum
import java.io.File
import java.util.*

// The data set for this demo can be found here: https://pjreddie.com/projects/mnist-in-csv/
fun main(args: Array<String>) {

    if (args.size != 2) {
        throw Exception("Please specify the paths to the MNIST training data and the test data (in the CSV format).")
    }

    val random = Random(1)

    val numberIterations = 30
    val batchSize = 1

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(args.first()), true)
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(args.last()), true)

    val inputDimension = 784
    val hiddenDimension = 100
    val numberCategories = MnistData.numberCategories

    val initialization = gaussianInitialization(random, 0.0f, 0.1f)
    val optimizer = momentum(0.005f, 0.1f)
    val keepProbability = 0.8f

    val network = Network(
        batchSize,
        input(inputDimension),
        dense(hiddenDimension, Activation.ReLU, initialization, optimizer),
        dropout(random, keepProbability),
        dense(numberCategories, Activation.Softmax, initialization, optimizer)
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
        crossEntropyLoss()) { _ : Int, _ : Float ->
            println(test.run())
        }
        .run()

}