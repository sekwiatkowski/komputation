package shape.komputation.network

import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.entry.EntryPoint
import shape.komputation.layers.entry.OptimizableEntryPoint
import shape.komputation.loss.LossFunction
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

val printLoss = { _ : Int, loss : Double ->println(loss) }

class Network(private val entryPoint: EntryPoint, private vararg val layers: FeedForwardLayer) {

    private val numberLayers = layers.size
    private val optimizableLayers = layers.filterIsInstance(OptimizableLayer::class.java).reversed()

    fun forward(input : Matrix) : RealMatrix {

        var output = entryPoint.forward(input)

        for (continuationLayer in layers) {

            output = continuationLayer.forward(output)

        }

        return output

    }

    fun backward(lossGradient: RealMatrix): RealMatrix {

        var chain = lossGradient

        for(indexLayer in numberLayers - 1 downTo 0) {

            chain = layers[indexLayer].backward(chain)

        }

        return chain

    }

    fun optimize(input : Matrix, endOfBackpropagation: RealMatrix) {

        optimizeContinuationLayers()

        if (entryPoint is OptimizableEntryPoint) {

            entryPoint.optimize(input, endOfBackpropagation)

        }

    }

    fun optimizeContinuationLayers() {

        for (layer in optimizableLayers) {

            layer.optimize()

        }

    }

    fun train(
        inputs: Array<Matrix>,
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

                val endOfBackpropagation = this.backward(lossGradient)

                this.optimize(input, endOfBackpropagation)

                iterationLoss += loss

            }

            afterEachIteration(indexIteration, iterationLoss)

        }


    }

}