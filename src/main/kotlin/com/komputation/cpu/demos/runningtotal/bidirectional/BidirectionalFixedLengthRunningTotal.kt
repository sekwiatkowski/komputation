package com.komputation.cpu.demos.runningtotal.bidirectional

import com.komputation.cpu.network.Network
import com.komputation.demos.runningtotal.RunningTotalData
import com.komputation.initialization.uniformInitialization
import com.komputation.instructions.entry.input
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.projection.weighting
import com.komputation.instructions.recurrent.ResultExtraction
import com.komputation.instructions.recurrent.bidirectionalRecurrentLayer
import com.komputation.loss.printLoss
import com.komputation.instructions.loss.squaredLoss
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
    val optimization = stochasticGradientDescent(0.001f)

    val steps = 2

    val numberExamples = 10_000
    val input = RunningTotalData.generateFixedLengthInput(random, steps, 0, 10, numberExamples)
    val forwardTargets = RunningTotalData.generateTargets(input)
    val backwardTargets = RunningTotalData.generateReversedTargets(input)
    val sumTargets = Array(numberExamples) { index ->
        val forwardTarget = forwardTargets[index]
        val backwardTarget = backwardTargets[index]

        forwardTarget
            .zip(backwardTarget).map { (a, b) -> a+b }
            .toFloatArray()
    }

    Network(
            1,
            input(1, steps),
            bidirectionalRecurrentLayer(
                1,
                Activation.Identity,
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