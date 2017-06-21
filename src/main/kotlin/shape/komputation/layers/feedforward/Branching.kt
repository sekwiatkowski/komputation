package shape.komputation.layers.feedforward

import shape.komputation.functions.concatRows
import shape.komputation.functions.splitRows
import shape.komputation.layers.ContinuationLayer
import shape.komputation.layers.OptimizableLayer
import shape.komputation.layers.entry.InputLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.EMPTY_DOUBLE_MATRIX
import shape.komputation.networks.Network

class Branching(name : String? = null, vararg continuations: Array<ContinuationLayer>) : ContinuationLayer(name), OptimizableLayer {

    private val networks = continuations.map { layers -> Network(InputLayer(), *layers) }

    private val results = Array(continuations.size) { EMPTY_DOUBLE_MATRIX }

    private val heights = IntArray(continuations.size)

    override fun forward(input : DoubleMatrix) : DoubleMatrix {

        for (indexNetwork in (0..networks.size-1)) {

            val network = networks[indexNetwork]

            val individualResult = network.forward(input)

            results[indexNetwork] = individualResult

            heights[indexNetwork] = individualResult.numberRows

        }

        return concatRows(*results)

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val chainSplit = splitRows(chain, this.heights)

        val resultWrtInput = this.networks.first().backward(chainSplit[0])

        val resultEntries = resultWrtInput.entries

        for (indexNetwork in (1..this.networks.size-1)) {

            val network = networks[indexNetwork]

            val remainingResultWrtInput = network.backward(chainSplit[indexNetwork])

            for (index in 0..resultEntries.size - 1) {

                resultEntries[index] += remainingResultWrtInput.entries[index]
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

fun createBranching(vararg continuations: Array<ContinuationLayer>): Branching {

    return createBranching(null, *continuations)
}

fun createBranching(name : String?, vararg continuations: Array<ContinuationLayer>): Branching {

    return Branching(name, *continuations)
}