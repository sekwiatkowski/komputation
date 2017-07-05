package shape.komputation.layers.forward

import shape.komputation.functions.splitRows
import shape.komputation.functions.stackRows
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.entry.inputLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.EMPTY_DOUBLE_MATRIX
import shape.komputation.networks.Network
import shape.komputation.optimization.Optimizable

class Concatenation internal constructor(name : String? = null, vararg continuations: Array<ForwardLayer>) : ForwardLayer(name), Optimizable {

    private val networks = continuations.map { layers -> Network(inputLayer(), *layers) }

    private val results = Array(continuations.size) { EMPTY_DOUBLE_MATRIX }

    private val heights = IntArray(continuations.size)

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        for (indexNetwork in (0..this.networks.size-1)) {

            val network = this.networks[indexNetwork]

            val individualResult = network.forward(input, isTraining)

            this.results[indexNetwork] = individualResult

            this.heights[indexNetwork] = individualResult.numberRows

        }

        val stackedResults = stackRows(this.results.first().numberColumns, *this.results)

        return stackedResults

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainSplit = splitRows(chain, this.heights)

        val firstNetwork = this.networks.first()
        val firstChainPart = chainSplit[0]

        val resultWrtInput = firstNetwork.backward(firstChainPart)

        val resultEntries = resultWrtInput.entries

        for (indexNetwork in (1..this.networks.size-1)) {

            val network = this.networks[indexNetwork]

            val remainingResultWrtInput = network.backward(chainSplit[indexNetwork])

            for (index in 0..resultEntries.size - 1) {

                resultEntries[index] += remainingResultWrtInput.entries[index]
            }

        }

        return resultWrtInput

    }

    override fun optimize() {

        for (network in this.networks) {

            network.optimize()

        }

    }

}

fun concatenation(vararg continuations: Array<ForwardLayer>) =

    concatenation(null, *continuations)

fun concatenation(vararg continuations: ForwardLayer) =

    concatenation(null, *continuations)

fun concatenation(name : String?, vararg continuations: ForwardLayer) =

    concatenation(name, *continuations.map { layer -> arrayOf<ForwardLayer>(layer) }.toTypedArray())

fun concatenation(name : String?, vararg continuations: Array<ForwardLayer>) =

    Concatenation(name, *continuations)