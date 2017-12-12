package com.komputation.cpu.demos.total

import com.komputation.cpu.layers.recurrent.Direction
import com.komputation.cpu.network.Network
import com.komputation.demos.total.TotalData
import com.komputation.initialization.zeroInitialization
import com.komputation.layers.entry.inputLayer
import com.komputation.layers.forward.activation.ActivationFunction
import com.komputation.layers.recurrent.ResultExtraction
import com.komputation.layers.recurrent.recurrentLayer
import com.komputation.loss.printLoss
import com.komputation.loss.squaredLoss
import com.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)

    val initialization = zeroInitialization()
    val optimization = stochasticGradientDescent(0.01f)

    val steps = 2

    val input = TotalData.generateFixedLengthInput(random, steps, 0, 10, 10_000)
    val targets = TotalData.generateTargets(input)

    Network(
            1,
            inputLayer(1, steps),
            recurrentLayer(
                steps,
                true,
                1,
                1,
                ActivationFunction.Identity,
                ResultExtraction.LastStep,
                Direction.Forward,
                initialization,
                optimization)
        )
        .training(
            input,
            targets,
            2,
            squaredLoss(1, 1, true),
            printLoss
        )
        .run()

}