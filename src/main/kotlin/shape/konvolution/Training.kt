package shape.konvolution

import shape.konvolution.loss.LossFunction

fun train(network: Network, inputs: Array<Matrix>, targets: Array<RealMatrix>, lossFunction: LossFunction, numberIterations : Int) {

    repeat(numberIterations) {

        var iterationLoss = 0.0

        inputs.zip(targets).forEach { (input, target) ->

            val forwardResults = network.forward(input)

            val prediction = forwardResults.last()

            val loss = lossFunction.forward(prediction, target)

            val lossGradient = lossFunction.backward(prediction, target)

            val backwardResults = network.backward(forwardResults, lossGradient)

            network.optimize(backwardResults)

            iterationLoss += loss

        }

        println(iterationLoss)

    }


}