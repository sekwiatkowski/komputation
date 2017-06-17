package shape.komputation.layers.feedforward

import shape.komputation.functions.concatRows
import shape.komputation.functions.splitRows
import shape.komputation.layers.FeedForwardLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.EMPTY_DOUBLE_MATRIX
import shape.komputation.networks.Network

class Concatenation(name : String? = null, vararg continuations: Array<FeedForwardLayer>) : FeedForwardLayer(name), OptimizableLayer {

    val networks = continuations.map { layers -> Network(InputLayer(), *layers) }

    val individualResults = Array(continuations.size) { EMPTY_DOUBLE_MATRIX }

    val individualHeights = IntArray(continuations.size)

    var input : DoubleMatrix? = null
    var forwardResult : DoubleMatrix? = null

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        this.input = input

        for (indexNetwork in (0..networks.size-1)) {

            val network = networks[indexNetwork]

            val individualResult = network.forward(input)

            individualResults[indexNetwork] = individualResult

            individualHeights[indexNetwork] = individualResult.numberRows

        }

        this.forwardResult = concatRows(*individualResults)

        return this.forwardResult!!

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainSplit = splitRows(chain, this.individualHeights)

        val resultWrtInput = this.networks.first().backward(chainSplit[0])

        val resultEntries = resultWrtInput.entries

        for (indexNetwork in (1..this.networks.size-1)) {

            val network = networks[indexNetwork]

            val secondResultWrtInput = network.backward(chainSplit[indexNetwork])

            for (index in 0..resultEntries.size - 1) {

                resultEntries[index] += secondResultWrtInput.entries[index]
            }


        }

        return resultWrtInput

    }

    override fun optimize() {

        for (network in networks) {

            network.optimize()

        }

    }

}

fun createConcatenation(vararg continuations: Array<FeedForwardLayer>): Concatenation {

    return createConcatenation(null, *continuations)
}

fun createConcatenation(name : String?, vararg continuations: Array<FeedForwardLayer>): Concatenation {

    return Concatenation(name, *continuations)
}