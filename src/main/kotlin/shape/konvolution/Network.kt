package shape.konvolution

import no.uib.cipr.matrix.Matrix
import shape.konvolution.layers.Layer
import shape.konvolution.optimization.Optimizable

class Network(private val layers : Array<Layer>) {

    val numberLayers = layers.size

    fun forward(input : Matrix): Array<Matrix> {

        var previousResult : Matrix = input

        val results = Array(layers.size + 1) { indexLayer ->

            if (indexLayer == 0) {

                previousResult = input

            }
            else {

                val layer = layers[indexLayer - 1]

                previousResult = layer.forward(previousResult)

            }

            previousResult

        }

        return results

    }

    fun backward(forwardResults : Array<Matrix>, lossGradient: Matrix) : Array<BackwardResult> {

        var chain = lossGradient

        return Array(numberLayers) { indexLayer ->

            val reverseIndex = numberLayers - indexLayer - 1

            val layer = layers[reverseIndex]

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

            val layer = layers[numberLayers-1-index]

            if (layer is Optimizable) {

                layer.optimize(backwardResults[index].parameter!!)
            }

        }

    }

}