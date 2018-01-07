package com.komputation.cpu.demos.increment

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.network
import com.komputation.demos.increment.IncrementData
import com.komputation.initialization.providedInitialization
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

    val oneInitialization = providedInitialization(floatArrayOf(1f), 1)
    val zeroInitialization = zeroInitialization()
    val optimization = stochasticGradientDescent(0.001f)

    val steps = 2

    val input = IncrementData.generateInput(random, steps, 0, 10, 10_000)
    val targets = IncrementData.generateTargets(input, 2)

    network(
        1,
        input(1, steps),
        recurrent(1, RecurrentActivation.Identity, ResultExtraction.AllSteps, Direction.LeftToRight, zeroInitialization, zeroInitialization, zeroInitialization, optimization),
        recurrent(1, RecurrentActivation.Identity, ResultExtraction.AllSteps, Direction.LeftToRight, oneInitialization, zeroInitialization, oneInitialization, null)
        ).training(
            input,
            targets,
            2,
        squaredLoss(),
            printLoss
        )
        .run()

}