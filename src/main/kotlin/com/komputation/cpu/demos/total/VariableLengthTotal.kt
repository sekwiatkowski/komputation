package com.komputation.cpu.demos.total

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.network
import com.komputation.demos.total.TotalData
import com.komputation.initialization.zeroInitialization
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.squaredLoss
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.recurrent
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = zeroInitialization()
    val optimization = stochasticGradientDescent(0.001f)

    val minimumLength = 2
    val maximumLength = 5

    val input = TotalData.generateVariableLengthInput(random, minimumLength, maximumLength, 0, 10, 10_000)
    val targets = TotalData.generateTargets(input)

    val network = network(
        1,
        input(1, minimumLength, maximumLength),
        recurrent(
            1,
            RecurrentActivation.Identity,
            ResultExtraction.LastStep,
            Direction.LeftToRight,
            initialization,
            initialization,
            null,
            optimization)
    )
    network
        .training(
            input,
            targets,
            2,
            squaredLoss(),
            printLoss
        )
        .run()

}