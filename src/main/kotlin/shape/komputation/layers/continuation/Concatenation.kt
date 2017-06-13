package shape.komputation.layers.continuation

import shape.komputation.Network
import shape.komputation.functions.concatRows
import shape.komputation.functions.splitRows
import shape.komputation.layers.entry.InputLayer
import shape.komputation.matrix.EMPTY_MATRIX
import shape.komputation.matrix.RealMatrix

class Concatenation(name : String? = null, vararg continuations: Array<ContinuationLayer>) : ContinuationLayer(name), OptimizableContinuationLayer {

    val networks = continuations.map { layers -> Network(InputLayer(), *layers) }

    val individualResults = Array(continuations.size) { EMPTY_MATRIX }

    val individualHeights = IntArray(continuations.size)

    var input : RealMatrix? = null
    var forwardResult : RealMatrix? = null

    override fun forward(input : RealMatrix) : RealMatrix {

        this.input = input

        for (indexNetwork in (0..networks.size-1)) {

            val network = networks[indexNetwork]

            val individualResult = network.forward(input)

            individualResults[indexNetwork] = individualResult

            individualHeights[indexNetwork] = individualResult.numberRows()

        }

        this.forwardResult = concatRows(*individualResults)

        return this.forwardResult!!

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(chain : RealMatrix) : RealMatrix {

        val chainSplit = splitRows(chain, this.individualHeights)

        var resultWrtInput = this.networks.first().backward(chainSplit[0])

        for (indexNetwork in (1..this.networks.size-1)) {

            val network = networks[indexNetwork]

            val secondResultWrtInput = network.backward(chainSplit[indexNetwork])

            resultWrtInput = resultWrtInput.add(secondResultWrtInput)

        }

        return resultWrtInput

    }

    override fun optimize() {

        for (network in networks) {

            network.optimizeContinuationLayers()

        }

    }

}

fun createConcatenation(vararg continuations: Array<ContinuationLayer>): Concatenation {

    return createConcatenation(null, *continuations)
}

fun createConcatenation(name : String?, vararg continuations: Array<ContinuationLayer>): Concatenation {

    return Concatenation(name, *continuations)
}
