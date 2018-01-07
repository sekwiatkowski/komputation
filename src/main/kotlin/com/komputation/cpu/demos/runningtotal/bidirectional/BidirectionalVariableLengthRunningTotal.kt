package com.komputation.cpu.demos.runningtotal.bidirectional

import com.komputation.cpu.network.network
import com.komputation.demos.runningtotal.RunningTotalData
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.continuation.activation.RecurrentActivation
import com.komputation.instructions.continuation.projection.weighting
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.squaredLoss
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.bidirectionalRecurrent
import com.komputation.loss.printLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

/*
    Input       1 2 3
    Forward     1 3 6
    Backward    6 5 3
    Sum         7 8 9
 */

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = uniformInitialization(random, -0.01f, 0.01f)
    val optimization = stochasticGradientDescent(0.0005f)

    val minimumLength = 2
    val maximumLength = 5

    val numberExamples = 10_000
    val input = RunningTotalData.generateVariableLengthInput(random, minimumLength, maximumLength, 0, 10, numberExamples)
    val forwardTargets = RunningTotalData.generateTargets(input)
    val backwardTargets = RunningTotalData.generateReversedTargets(input)
    val sumTargets = Array(numberExamples) { index ->
        val forwardTarget = forwardTargets[index]
        val backwardTarget = backwardTargets[index]

        forwardTarget
            .zip(backwardTarget).map { (a, b) -> a+b }
            .toFloatArray()
    }

    network(
            1,
            input(1, minimumLength, maximumLength),
            bidirectionalRecurrent(
                1,
                RecurrentActivation.Identity,
                ResultExtraction.AllSteps,
                initialization,
                optimization
            ),
            weighting(1, initialization, optimization)
        )
        .training(
            input,
            sumTargets,
            2,
            squaredLoss(),
            printLoss
        )
        .run()

}