package com.komputation.cpu.demos.total

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.network
import com.komputation.demos.total.TotalData
import com.komputation.initialization.zeroInitialization
import com.komputation.instructions.continuation.activation.Activation
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
    val optimization = stochasticGradientDescent(0.01f)

    val steps = 2

    val input = TotalData.generateFixedLengthInput(random, steps, 0, 10, 10_000)
    val targets = TotalData.generateTargets(input)

    network(
            1,
            input(1, steps),
            recurrent(
                1,
                Activation.Identity,
                ResultExtraction.LastStep,
                Direction.LeftToRight,
                initialization,
                initialization,
                null,
                optimization)
        )
        .training(
            input,
            targets,
            2,
            squaredLoss(),
            printLoss
        )
        .run()

}