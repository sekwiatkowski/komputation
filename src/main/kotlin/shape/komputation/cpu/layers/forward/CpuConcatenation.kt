package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.Network
import shape.komputation.cpu.functions.splitRows
import shape.komputation.cpu.functions.stackRows
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.entry.inputLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.EMPTY_DOUBLE_MATRIX
import shape.komputation.optimization.Optimizable

class CpuConcatenation internal constructor(name : String? = null, inputDimension : Int, continuations: Array<Array<CpuForwardLayerInstruction>>) : BaseCpuForwardLayer(name), Optimizable {

    private val networks = continuations.map { layers -> Network(inputLayer(inputDimension), *layers) }
    private val numberNetworks = this.networks.size

    private val results = Array(this.numberNetworks) { EMPTY_DOUBLE_MATRIX }

    private val heights = IntArray(this.numberNetworks)

    override fun forward(input : DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        for (indexNetwork in (0..this.numberNetworks-1)) {

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

        for (indexNetwork in (1..this.numberNetworks-1)) {

            val network = this.networks[indexNetwork]

            val remainingResultWrtInput = network.backward(chainSplit[indexNetwork])

            for (index in 0..resultEntries.size - 1) {

                resultEntries[index] += remainingResultWrtInput.entries[index]
            }

        }

        return resultWrtInput

    }

    override fun optimize(scalingFactor : Double) {

        for (network in this.networks) {

            network.optimize(scalingFactor)

        }

    }

}