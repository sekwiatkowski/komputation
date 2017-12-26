package com.komputation.cpu.demos.runningtotal.lefttoright

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.network
import com.komputation.demos.runningtotal.RunningTotalData
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
    val optimization = stochasticGradientDescent(0.001f)

    val minimumLength = 2
    val maximumLength = 5

    val input = RunningTotalData.generateVariableLengthInput(random, minimumLength, maximumLength, 0, 10, 10_000)
    val targets = RunningTotalData.generateTargets(input)

    network(
            1,
            input(1, minimumLength, maximumLength),
            recurrent(
                1,
                Activation.Identity,
                ResultExtraction.AllSteps,
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