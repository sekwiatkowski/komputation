package shape.komputation.demos

import shape.komputation.*
import shape.komputation.layers.entry.InputLayer
import shape.komputation.layers.continuation.activation.SigmoidLayer
import shape.komputation.layers.continuation.createProjectionLayer
import shape.komputation.loss.SquaredLoss
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import shape.komputation.optimization.stochasticGradientDescent
import java.util.*

fun main(args: Array<String>) {

    val input = arrayOf<Matrix>(
        createRealMatrix(doubleArrayOf(0.0), doubleArrayOf(0.0)),
        createRealMatrix(doubleArrayOf(1.0), doubleArrayOf(0.0)),
        createRealMatrix(doubleArrayOf(0.0), doubleArrayOf(1.0)),
        createRealMatrix(doubleArrayOf(1.0), doubleArrayOf(1.0))
    )

    val targets = arrayOf(
        createRealMatrix(doubleArrayOf(0.0)),
        createRealMatrix(doubleArrayOf(1.0)),
        createRealMatrix(doubleArrayOf(1.0)),
        createRealMatrix(doubleArrayOf(0.0))
    )

    val random = Random(1)
    val initialize = createUniformInitializer(random, -0.5, 0.5)

    val inputLayer = InputLayer()

    val updateRule = stochasticGradientDescent(0.1)

    val hiddenPreactivationLayer = createProjectionLayer(2, 2, initialize, updateRule)
    val hiddenLayer = SigmoidLayer()

    val outputPreactivationLayer = createProjectionLayer(2, 1, initialize, updateRule)
    val outputLayer = SigmoidLayer()

    val network = Network(
        inputLayer,
        hiddenPreactivationLayer,
        hiddenLayer,
        outputPreactivationLayer,
        outputLayer
    )

    train(network, input, targets, SquaredLoss(), 30_000, printLoss)

}