package com.komputation.cpu.layers.forward

import com.komputation.cpu.functions.splitRows
import com.komputation.cpu.functions.stackRows
import com.komputation.cpu.layers.CpuForwardLayer
import com.komputation.cpu.layers.VariableLengthFloatArray
import com.komputation.cpu.layers.computeLengthIndex
import com.komputation.optimization.Optimizable

class CpuConcatenation internal constructor(
    val name : String? = null,
    private val layers: Array<CpuForwardLayer>) : CpuForwardLayer, Optimizable {

    private val firstLayer = this.layers.first()
    private val numberLayers = this.layers.size

    // All layers have the same number of input rows.
    override val numberInputRows = this.firstLayer.numberInputRows
    // The number of input columns is flexible.
    override var numberInputColumns = -1
    // The possible input lengths are the same for all layers.
    override val possibleInputLengths: IntArray
        get() = this.firstLayer.possibleInputLengths

    // Layers can have different numbers of output rows.
    private val numbersOfOutputRows = IntArray(this.numberLayers) { index -> this.layers[index].numberOutputRows }
    // The number of output rows of the concatenation layer is the sum of the number of output rows of its layers.
    override val numberOutputRows = this.numbersOfOutputRows.sum()
    // The number of output columns is flexible.
    override var numberOutputColumns = -1
    // The possible output lengths are the same for all layers.
    override val possibleOutputLengths: IntArray
        get() = this.firstLayer.possibleOutputLengths
    private val minimumOutputLength = this.possibleOutputLengths.min()!!

    private val forwardStore = VariableLengthFloatArray(this.numberOutputRows, this.possibleOutputLengths)
    override var forwardResult = FloatArray(0)

    private val backwardStore = VariableLengthFloatArray(this.numberInputRows, this.possibleInputLengths)
    override var backwardResult  = FloatArray(0)

    private val individualResults = Array(this.numberLayers) { FloatArray(0) }

    override fun forward(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean): FloatArray {
        this.individualResults[0] = this.firstLayer.forward(withinBatch, numberInputColumns, input, isTraining)

        this.numberInputColumns = this.firstLayer.numberInputColumns
        this.numberOutputColumns = this.firstLayer.numberOutputColumns
        this.forwardResult = this.forwardStore.get(this.numberOutputColumns)

        for (indexLayer in (1 until this.numberLayers)) {
            this.individualResults[indexLayer] = this.layers[indexLayer].forward(withinBatch, numberInputColumns, input, isTraining)
        }

        stackRows(this.numbersOfOutputRows, this.numberOutputRows, this.numberOutputColumns, this.forwardResult, *this.individualResults)

        return this.forwardResult
    }

    private val chainSplitsOverPossibleLengths = this.possibleOutputLengths.map { outputLength ->
        Array(this.numberLayers) { indexLayer -> FloatArray(this.layers[indexLayer].numberOutputRows * outputLength) }
    }

    /*
        a b
        ---
        c d
    */
    override fun backward(withinBatch: Int, chain: FloatArray): FloatArray {
        this.backwardResult = this.backwardStore.get(this.numberInputColumns)

        val outputLengthIndex = computeLengthIndex(this.numberOutputColumns, this.minimumOutputLength)
        val chainSplits = this.chainSplitsOverPossibleLengths[outputLengthIndex]

        splitRows(this.numberOutputRows, this.numberOutputColumns, chain, this.numbersOfOutputRows, this.numberLayers, chainSplits)

        val firstIndividualBackwardResult = this.firstLayer.backward(withinBatch, chainSplits[0])
        System.arraycopy(firstIndividualBackwardResult, 0, this.backwardResult, 0, firstIndividualBackwardResult.size)

        for (indexLayer in (1 until this.numberLayers)) {
            val individualBackwardResult = this.layers[indexLayer].backward(withinBatch, chainSplits[indexLayer])

            for (index in 0 until individualBackwardResult.size) {
                this.backwardResult[index] += individualBackwardResult[index]
            }
        }

        return this.backwardResult
    }

    override fun optimize(batchSize : Int) {
        for (layer in this.layers) {
            if (layer is Optimizable) {
                layer.optimize(batchSize)
            }
        }
    }

}