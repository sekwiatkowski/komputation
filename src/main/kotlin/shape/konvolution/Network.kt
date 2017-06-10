package shape.konvolution

import shape.konvolution.layers.continuation.ContinuationLayer
import shape.konvolution.layers.entry.EntryPoint
import shape.konvolution.matrix.Matrix
import shape.konvolution.matrix.RealMatrix
import shape.konvolution.layers.continuation.OptimizableContinuationLayer
import shape.konvolution.layers.entry.OptimizableEntryPoint

class Network(private val entryPoint: EntryPoint, private vararg val continuationLayers: ContinuationLayer) {

    private val numberContinuationLayers = continuationLayers.size

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

        val results = Array(numberContinuationLayers) { indexLayer ->

            val reverseIndex = numberContinuationLayers - indexLayer - 1

            val layer = continuationLayers[reverseIndex]

            val input = forwardResults[reverseIndex]
            val output = forwardResults[reverseIndex + 1]

            val backwardResult = layer.backward(input, output, chain)

            val (inputGradient, _) = backwardResult

            chain = inputGradient

            backwardResult

        }

        return results

    }

    fun optimize(input: Matrix, forwardResults : Array<RealMatrix>, backwardResults: Array<BackwardResult>) {

        optimizeContinuationLayers(backwardResults)

        if (entryPoint is OptimizableEntryPoint) {

            entryPoint.optimize(input, forwardResults.first(), backwardResults.last().input)

        }

    }

    private fun optimizeContinuationLayers(backwardResults: Array<BackwardResult>) {

        for (index in 0..numberContinuationLayers - 1) {

            val layer = continuationLayers[numberContinuationLayers - 1 - index]

            if (layer is OptimizableContinuationLayer) {

                val parameterGradients = backwardResults[index]

                layer.optimize(parameterGradients.parameter!!)

            }

        }

    }

}