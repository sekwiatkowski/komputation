package com.komputation.cpu.demos.runningtotal.bidirectional

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.Network
import com.komputation.demos.runningtotal.RunningTotalData
import com.komputation.initialization.uniformInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.forward.concatenation
import com.komputation.layers.forward.projection.weightingLayer
import com.komputation.layers.recurrent.ResultExtraction
import com.komputation.layers.recurrent.recurrentLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
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

    val hasFixedLength = true

    Network(
            1,
            inputLayer(1, steps),
            concatenation(
                recurrentLayer(steps, hasFixedLength, 1, 1, Direction.Forward, ResultExtraction.AllSteps, ActivationFunction.Identity, initialization, initialization, null, optimization),
                recurrentLayer(steps, hasFixedLength, 1, 1, Direction.Backward, ResultExtraction.AllSteps, ActivationFunction.Identity, initialization, initialization, null, optimization)
            ),
            weightingLayer(2, steps, true, 1, initialization, optimization)
        )
        .training(
            input,
            sumTargets,
            2,
            squaredLoss(1, steps, hasFixedLength),
            printLoss
        )
        .run()

}