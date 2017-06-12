package shape.konvolution.layers.continuation

import shape.konvolution.Network
import shape.konvolution.functions.concatRows
import shape.konvolution.functions.splitRows
import shape.konvolution.layers.entry.InputLayer
import shape.konvolution.matrix.EMPTY_MATRIX
import shape.konvolution.matrix.RealMatrix

class Concatenation(name : String? = null, vararg continuations: Array<ContinuationLayer>) : ContinuationLayer(name, 1, 0), OptimizableContinuationLayer {

    val networks = continuations.map { layers -> Network(InputLayer(), *layers) }

    val individualResults = Array(continuations.size) { EMPTY_MATRIX }

    val individualHeights = IntArray(continuations.size)

    override fun forward() {

        val input = this.lastInput!!

        for (indexNetwork in (0..networks.size-1)) {

            val network = networks[indexNetwork]

            val individualResult = network.forward(input)

            individualResults[indexNetwork] = individualResult

            individualHeights[indexNetwork] = individualResult.numberRows()

        }

        val concatenation = concatRows(*individualResults)

        this.lastForwardResult[0] = concatenation

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(chain : RealMatrix) {

        val chainSplit = splitRows(chain, this.individualHeights)

        var resultWrtInput = this.networks.first().backward(chainSplit[0])

        for (indexNetwork in (1..this.networks.size-1)) {

            val network = networks[indexNetwork]

            val secondResultWrtInput = network.backward(chainSplit[indexNetwork])

            resultWrtInput = resultWrtInput.add(secondResultWrtInput)

        }

        this.lastBackwardResultWrtInput = resultWrtInput

    }

    override fun optimize() {

        for (network in networks) {

            network.optimize()

        }

    }

}

fun createConcatenation(vararg continuations: Array<ContinuationLayer>): Concatenation {

    return createConcatenation(null, *continuations)
}

fun createConcatenation(name : String?, vararg continuations: Array<ContinuationLayer>): Concatenation {

    return Concatenation(name, *continuations)
}
