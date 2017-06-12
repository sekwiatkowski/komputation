package shape.komputation

import shape.komputation.layers.continuation.ContinuationLayer
import shape.komputation.layers.entry.EntryPoint
import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix
import shape.komputation.layers.continuation.OptimizableContinuationLayer
import shape.komputation.layers.entry.OptimizableEntryPoint

class Network(private val entryPoint: EntryPoint, private vararg val continuationLayers: ContinuationLayer) {

    private val numberContinuationLayers = continuationLayers.size

    fun forward(input : Matrix) : RealMatrix {

        for (indexLayer in 0..continuationLayers.size) {

            if (indexLayer == 0) {

                entryPoint.run {
                    setInput(input)
                    forward()
                }

            }
            else {

                val previousOutput = if(indexLayer == 1) entryPoint.lastForwardResult!! else continuationLayers[indexLayer - 1 - 1].lastForwardResult.last()

                val layer = continuationLayers[indexLayer - 1]

                layer.run {
                    setInput(previousOutput)
                    forward()
                }

            }
        }

        val output = continuationLayers.last().lastForwardResult.last()

        return output

    }

    fun backward(lossGradient: RealMatrix): RealMatrix {

        var chain = lossGradient

        for(indexLayer in 0..numberContinuationLayers-1) {

            val reverseIndex = numberContinuationLayers - indexLayer - 1

            val layer = continuationLayers[reverseIndex]

            layer.backward(chain)

            chain = layer.lastBackwardResultWrtInput!!

        }

        return chain

    }

    fun optimize() {

        optimizeContinuationLayers()

        if (entryPoint is OptimizableEntryPoint) {

            val firstContinuationLayer = this.continuationLayers.first()

            entryPoint.optimize(firstContinuationLayer.lastBackwardResultWrtInput!!)

        }

    }

    private fun optimizeContinuationLayers() {

        for (index in 0..numberContinuationLayers - 1) {

            val layer = continuationLayers[numberContinuationLayers - 1 - index]

            if (layer is OptimizableContinuationLayer) {

                layer.optimize()

            }

        }

    }

}