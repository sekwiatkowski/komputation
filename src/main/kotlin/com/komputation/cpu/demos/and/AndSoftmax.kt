package com.komputation.cpu.demos.and

import com.komputation.cpu.network.network
import com.komputation.demos.and.OneHotAndData
import com.komputation.initialization.heInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.crossEntropyLoss
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val inputDimension = 2
    val outputDimension = 2
    val batchSize = 4

    val random = Random(1)
    val initialization = heInitialization(random)

    val optimization = stochasticGradientDescent(0.03f)

    network(
        batchSize,
        input(inputDimension),
        dense(outputDimension, Activation.Softmax, initialization, optimization)
    )
        .training(
            OneHotAndData.input,
            OneHotAndData.targets,
            10_000,
            crossEntropyLoss(),
            printLoss)
        .run()

}