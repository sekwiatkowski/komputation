package shape.konvolution

import shape.konvolution.layers.continuation.ContinuationLayer
import shape.konvolution.layers.entry.EntryPoint
import shape.konvolution.optimization.Optimizable

class Network(private val entryPoint: EntryPoint, vararg val continuationLayers: ContinuationLayer) {

    val numberLayers = continuationLayers.size

    fun forward(input : Matrix): Array<RealMatrix> {

        var previousResult : RealMatrix? = null

        val results = Array(continuationLayers.size + 1) { indexLayer ->

            if (indexLayer == 0) {

                previousResult = entryPoint.forward(input)

            }
            else {

                val layer = continuationLayers[indexLayer - 1]

                previousResult = layer.forward(previousResult!!)

            }

            previousResult!!

        }

        return results

    }

    fun backward(forwardResults : Array<RealMatrix>, lossGradient: RealMatrix) : Array<BackwardResult> {

        var chain = lossGradient

        return Array(numberLayers) { indexLayer ->

            val reverseIndex = numberLayers - indexLayer - 1

            val layer = continuationLayers[reverseIndex]

            val input = forwardResults[reverseIndex]
            val output = forwardResults[reverseIndex + 1]

            val backwardResult = layer.backward(input, output, chain)

            val (inputGradient, _) = backwardResult

            chain = inputGradient

            backwardResult

        }

    }

    fun optimize(backwardResults: Array<BackwardResult>) {

        for (index in 0..numberLayers-1) {

            val layer = continuationLayers[numberLayers-1-index]

            if (layer is Optimizable) {

                layer.optimize(backwardResults[index].parameter!!)
            }

        }

    }

}