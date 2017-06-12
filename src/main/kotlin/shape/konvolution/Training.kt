package shape.konvolution

import shape.konvolution.loss.LossFunction
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix

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

            network.backward(lossGradient)

            network.optimize()

            iterationLoss += loss

        }

        afterEachIteration(indexIteration, iterationLoss)

    }


}