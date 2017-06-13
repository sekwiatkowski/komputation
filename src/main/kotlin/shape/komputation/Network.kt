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

        var output = entryPoint.forward(input)

        for (continuationLayer in continuationLayers) {

            output = continuationLayer.forward(output)

        }

        return output

    }

    fun backward(lossGradient: RealMatrix): RealMatrix {

        var chain = lossGradient

        for(indexLayer in numberContinuationLayers-1 downTo 0) {

            chain = continuationLayers[indexLayer].backward(chain)

        }

        return chain

    }

    fun optimize(endOfBackpropagation: RealMatrix) {

        optimizeContinuationLayers()

        if (entryPoint is OptimizableEntryPoint) {

            entryPoint.optimize(endOfBackpropagation)

        }

    }

    fun optimizeContinuationLayers() {

        for (index in 0..numberContinuationLayers - 1) {

            val layer = continuationLayers[numberContinuationLayers - 1 - index]

            if (layer is OptimizableContinuationLayer) {

                layer.optimize()

            }

        }

    }

}