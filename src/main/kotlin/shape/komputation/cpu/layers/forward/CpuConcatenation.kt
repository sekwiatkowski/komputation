package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.Network
import shape.komputation.cpu.functions.splitRows
import shape.komputation.cpu.functions.stackRows
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.entry.inputLayer
import shape.komputation.matrix.EMPTY_FLOAT_MATRIX
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuConcatenation internal constructor(
    name : String? = null,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    continuations: Array<Array<CpuForwardLayerInstruction>>) : BaseCpuForwardLayer(name), Optimizable {

    private val networks = continuations.map { layers -> Network(inputLayer(this.numberInputRows, this.numberInputColumns), *layers) }
    private val numberNetworks = this.networks.size

    private val results = Array(this.numberNetworks) { FloatArray(0) }
    private val heights = IntArray(this.numberNetworks) { -1 }

    override fun forward(withinBatch : Int, input : FloatMatrix, isTraining : Boolean) : FloatMatrix {

        var numberColumns = -1

        for (indexNetwork in (0..this.numberNetworks-1)) {

            val network = this.networks[indexNetwork]

            val individualResult = network.forward(withinBatch, input, isTraining)
            val individualResultEntries = individualResult.entries

            this.results[indexNetwork] = individualResultEntries
            this.heights[indexNetwork] = individualResult.numberRows

            numberColumns = individualResult.numberColumns

        }

        var totalNumberRows = 0
        for (height in heights) {
            totalNumberRows += height
        }

        val stacked = FloatArray(numberColumns * totalNumberRows)

        stackRows(this.heights, totalNumberRows, numberColumns, stacked, *this.results)

        return FloatMatrix(totalNumberRows, numberColumns, stacked)

    }

    // Chain is the same for (1, 2) and (2, 1)
    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        val chainSplit = splitRows(chain, this.heights)

        val firstNetwork = this.networks.first()
        val firstChainPart = chainSplit[0]

        val resultWrtInput = firstNetwork.backward(withinBatch, firstChainPart)

        val resultEntries = resultWrtInput.entries

        for (indexNetwork in (1..this.numberNetworks-1)) {

            val network = this.networks[indexNetwork]

            val remainingResultWrtInput = network.backward(withinBatch, chainSplit[indexNetwork])

            for (index in 0..resultEntries.size - 1) {

                resultEntries[index] += remainingResultWrtInput.entries[index]
            }

        }

        return resultWrtInput

    }

    override fun optimize(scalingFactor : Float) {

        for (network in this.networks) {

            network.optimize(scalingFactor)

        }

    }

}