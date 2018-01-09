package com.komputation.cpu.demos.and

import com.komputation.cpu.network.network
import com.komputation.demos.and.BinaryAndData
import com.komputation.initialization.heInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.logisticLoss
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 1
    val batchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.1f)

    network(
        batchSize,
        input(inputDimension),
        dense(outputDimension, Activation.Sigmoid, initialization, optimization)
    )
        .training(
            BinaryAndData.inputs,
            BinaryAndData.targets,
            10_000,
            logisticLoss(),
            printLoss)
        .run()

}