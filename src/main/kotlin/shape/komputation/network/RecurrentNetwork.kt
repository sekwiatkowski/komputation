package shape.komputation.network

import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.RecurrentLayer
import shape.komputation.layers.entry.EntryPoint
import shape.komputation.layers.entry.OptimizableEntryPoint
import shape.komputation.loss.LossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

class RecurrentNetwork(
    maximumSteps : Int,
    private val entryPoint: EntryPoint,
    private vararg val layers: ContinuationLayer) {

    private val inputGradients = arrayOfNulls<RealMatrix>(maximumSteps)
    private val numberLayers = layers.size
    private val optimizableLayers = layers.filterIsInstance(OptimizableLayer::class.java).reversed()

    fun forward(inputs : Array<Matrix>) : RealMatrix {

        var output : RealMatrix? = null

        val firstLayer = layers.first()

        if (firstLayer is RecurrentLayer) {

            firstLayer.resetForward()

        }

        for (input in inputs) {

            val entry = entryPoint.forward(input)

            output = firstLayer.forward(entry)

        }

        for (indexLayer in 1..numberLayers - 1) {

            val layer = layers[indexLayer]

            output = layer.forward(output!!)

        }

        return output!!

    }

    fun backward(numberSteps : Int, lossGradient: RealMatrix) {

        var chain = lossGradient

        for (indexLayer in numberLayers - 1 downTo 1) {

            val layer = layers[indexLayer]

            when (layer) {

                is FeedForwardLayer -> {

                    chain = layer.backward(chain)

                }
                is RecurrentLayer -> {

                    throw NotImplementedError()

                }

            }

        }

        val firstLayer = layers.first()

        when (firstLayer) {

            is RecurrentLayer -> {

                firstLayer.resetBackward()

                for (step in numberSteps - 1 downTo 0) {

                    val (stateGradient, inputGradient) = firstLayer.backward(chain)

                    chain = stateGradient

                    inputGradients[step] = inputGradient

                }

            }

            is FeedForwardLayer -> {

                for (step in numberSteps - 1 downTo 0) {

                    inputGradients[step] = firstLayer.backward(chain)

                }

            }

        }

    }

    fun optimize(inputs : Array<Matrix>) {

        for (layer in optimizableLayers) {

            layer.optimize()

        }

        if (entryPoint is OptimizableEntryPoint) {

            for (step in inputs.size downTo 0) {

                val input = inputs[step]
                val inputGradient = inputGradients[step]!!

                entryPoint.optimize(input, inputGradient)

            }

        }

    }

    fun train(
        inputs: Array<Array<Matrix>>,
        targets: Array<RealMatrix>,
        lossFunction: LossFunction,
        numberIterations : Int,
        afterEachIteration : (index : Int, loss : Double) -> Unit) {

        repeat(numberIterations) { indexIteration ->

            var iterationLoss = 0.0

            inputs.zip(targets).forEach { (input, target) ->

                val prediction = this.forward(input)

                val loss = lossFunction.forward(prediction, target)

                val lossGradient = lossFunction.backward(prediction, target)

                this.backward(input.size, lossGradient)

                this.optimize(input)

                iterationLoss += loss

            }

            afterEachIteration(indexIteration, iterationLoss)

        }

    }

}