package shape.komputation

import shape.komputation.loss.LossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

val printLoss = { _ : Int, loss : Double -> println(loss) }

fun train(
    network: Network,
    inputs: Array<Matrix>,
    targets: Array<RealMatrix>,
    lossFunction: LossFunction,
    numberIterations : Int,
    afterEachIteration : (index : Int, loss : Double) -> Unit) {

    repeat(numberIterations) { indexIteration ->

        var iterationLoss = 0.0

        inputs.zip(targets).forEach { (input, target) ->

            val forwardResults = network.forward(input)

            val prediction = forwardResults

            val loss = lossFunction.forward(prediction, target)

            val lossGradient = lossFunction.backward(prediction, target)

            val endOfBackpropagation = network.backward(lossGradient)

            network.optimize(endOfBackpropagation)

            iterationLoss += loss

        }

        afterEachIteration(indexIteration, iterationLoss)

    }


}