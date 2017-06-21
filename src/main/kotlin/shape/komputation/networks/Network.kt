package shape.komputation.networks

import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.entry.EntryPoint
import shape.komputation.loss.LossFunction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices

val printLoss = { _ : Int, loss : Double -> println(loss) }

class Network(private val entryPoint: EntryPoint, private vararg val layers: ContinuationLayer) {

    private val numberLayers = layers.size
    private val optimizables = listOf(entryPoint).plus(layers).filterIsInstance(OptimizableLayer::class.java).reversed()

    fun forward(input : Matrix) : DoubleMatrix {

        var output = entryPoint.forward(input)

        for (continuationLayer in layers) {

            output = continuationLayer.forward(output)

        }

        return output

    }

    fun backward(lossGradient : DoubleMatrix): DoubleMatrix {

        var chain = lossGradient

        for(indexLayer in numberLayers - 1 downTo 0) {

            val layer = layers[indexLayer]

            chain = layer.backward(chain)

        }

        return entryPoint.backward(chain)

    }

    fun optimize() {

        for (optimizable in optimizables) {

            optimizable.optimize()

        }

    }

    fun train(
        inputs: Array<Matrix>,
        targets: Array<DoubleMatrix>,
        lossFunction: LossFunction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Double) -> Unit)? = null) {

        val numberExamples = inputs.size

        val batches = partitionIndices(numberExamples, batchSize)

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0

            for (batch in batches) {

                var batchLoss = 0.0

                for (indexExample in batch) {

                    val input = inputs[indexExample]
                    val target = targets[indexExample]

                    val prediction = this.forward(input)

                    val loss = lossFunction.forward(prediction, target)

                    val lossGradient = lossFunction.backward(prediction, target)

                    this.backward(lossGradient)

                    batchLoss += loss

                }

                iterationLoss += batchLoss

                this.optimize()

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }


    }

}