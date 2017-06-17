package shape.komputation.networks

import shape.komputation.layers.*
import shape.komputation.layers.entry.InputLayer
import shape.komputation.loss.LossFunction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.Matrix

class RecurrentNetwork(
    private val inputLayer: InputLayer,
    private vararg val layers: ContinuationLayer) {

    private val numberLayers = layers.size
    private val optimizables = listOf(inputLayer).plus(layers).filterIsInstance(OptimizableLayer::class.java).reversed()

    fun forward(inputs : Array<Matrix>) : DoubleMatrix {

        val firstLayer = layers.first()

        if (firstLayer is StatefulLayer) {

            firstLayer.startForward()

        }

        var output : DoubleMatrix? = null

        for (input in inputs) {

            val entry = this.inputLayer.forward(input)

            output = firstLayer.forward(entry)

        }

        for (indexLayer in 1..numberLayers - 1) {

            val layer = this.layers[indexLayer]

            if (layer is StatefulLayer) {

                layer.startForward()
            }

            output = layer.forward(output!!)

        }

        return output!!

    }

    fun backward(numberSteps : Int, lossGradient: DoubleMatrix) {

        var chain = lossGradient

        for (indexLayer in numberLayers - 1 downTo 1) {

            val layer = this.layers[indexLayer]

            when (layer) {

                is FeedForwardLayer -> {

                    chain = layer.backward(chain)

                    if (layer is StatefulLayer) {

                        layer.finishBackward()

                    }

                }
                is RecurrentLayer -> {

                    throw NotImplementedError()

                }

            }

        }

        val firstLayer = layers.first()

        when (firstLayer) {

            is RecurrentLayer -> {

                for (step in numberSteps - 1 downTo 0) {

                    val (stateGradient, inputGradient) = firstLayer.backward(chain)

                    chain = stateGradient

                    inputLayer.backward(inputGradient)

                }

            }

            is FeedForwardLayer -> {

                throw NotImplementedError()
            }

        }

        if (firstLayer is StatefulLayer) {

            firstLayer.finishBackward()

        }

    }

    fun optimize() {

        for (layer in optimizables) {

            layer.optimize()

        }

    }

    fun train(
        inputs: Array<Array<Matrix>>,
        targets: Array<DoubleMatrix>,
        lossFunction: LossFunction,
        numberIterations : Int,
        batchSize : Int,
        afterEachIteration : ((index : Int, loss : Double) -> Unit)? = null) {

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0

            var indexBatch = 0

            inputs.zip(targets).forEach { (input, target) ->

                val prediction = this.forward(input)

                val loss = lossFunction.forward(prediction, target)

                val lossGradient = lossFunction.backward(prediction, target)

                this.backward(input.size, lossGradient)

                indexBatch++

                if (indexBatch == batchSize) {

                    this.optimize()
                    indexBatch = 0

                }

                iterationLoss += loss

            }

            if (afterEachIteration != null) {

                afterEachIteration(indexIteration, iterationLoss)

            }

        }

    }

}