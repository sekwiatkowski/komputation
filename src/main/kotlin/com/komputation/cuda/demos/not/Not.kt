package com.komputation.cuda.demos.not

import com.komputation.cuda.network.cudaNetwork
import com.komputation.demos.not.NotData
import com.komputation.initialization.heInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.logisticLoss
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 1
    val outputDimension = 1
    val batchSize = 2

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.1f)

    val network = cudaNetwork(
        batchSize,
        input(inputDimension),
        dense(outputDimension, Activation.Sigmoid, initialization, optimization)
    )

    val training = network
        .training(
            NotData.inputs,
            NotData.targets,
            10_000,
            logisticLoss(),
            printLoss)

    training.run()

    training.free()
    network.free()

}