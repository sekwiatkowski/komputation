package shape.komputation.networks

import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.Resourceful
import shape.komputation.layers.entry.EntryPoint
import shape.komputation.loss.LossFunction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.partitionIndices
import shape.komputation.optimization.Optimizable

val printLoss = { _ : Int, loss : Double -> println(loss) }

class Network(private val entryPoint: EntryPoint, private vararg val layers: ForwardLayer) {

    private val numberLayers = this.layers.size
    private val optimizables = listOf(this.entryPoint).plus(this.layers).filterIsInstance(Optimizable::class.java).reversed()

    fun forward(input : Matrix, isTraining : Boolean) : DoubleMatrix {

        var output = this.entryPoint.forward(input)

        for (layer in this.layers) {

            output = layer.forward(output, isTraining)

        }

        return output

    }

    fun backward(lossGradient : DoubleMatrix): DoubleMatrix {

        var chain = lossGradient

        for(indexLayer in this.numberLayers - 1 downTo 0) {

            val layer = this.layers[indexLayer]

            chain = layer.backward(chain)

        }

        return this.entryPoint.backward(chain)

    }

    fun optimize(scalingFactor : Double) {

        for (optimizable in this.optimizables) {

            optimizable.optimize(scalingFactor)

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

        this.acquireLayerResources()

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0

            for (batch in batches) {

                var batchLoss = 0.0

                for (indexExample in batch) {

                    val input = inputs[indexExample]
                    val target = targets[indexExample]

                    val prediction = this.forward(input, true)

                    val loss = lossFunction.forward(prediction, target)

                    val lossGradient = lossFunction.backward(prediction, target)

                    this.backward(lossGradient)

                    batchLoss += loss

                }

                iterationLoss += batchLoss

                val scalingFactor = 1.0.div(batch.size.toDouble())

                this.optimize(scalingFactor)

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

        this.releaseLayerResources()

    }

    fun test(
        inputs: Array<Matrix>,
        targets: Array<DoubleMatrix>,
        isCorrect: (DoubleMatrix, DoubleMatrix) -> Boolean) : BooleanArray {

        val size = inputs.size

        val results = BooleanArray(size)

        this.acquireLayerResources()

        for(index in 0..size -1) {

            val input = inputs[index]
            val target = targets[index]

            val prediction = this.forward(input, false)

            results[index] = isCorrect(prediction, target)

        }

        this.releaseLayerResources()

        return results

    }

    fun acquireLayerResources() {

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.acquire()

            }

        }

    }

    fun releaseLayerResources() {

        for (layer in this.layers) {

            if (layer is Resourceful) {

                layer.release()

            }

        }

    }

}