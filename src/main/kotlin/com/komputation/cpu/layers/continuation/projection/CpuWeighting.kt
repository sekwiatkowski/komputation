package com.komputation.cpu.layers.continuation.projection

import com.komputation.cpu.functions.backwardProjectionWrtInput
import com.komputation.cpu.functions.backwardProjectionWrtWeights
import com.komputation.cpu.functions.multiply
import com.komputation.cpu.layers.BaseCpuContinuationLayer
import com.komputation.cpu.layers.computeLengthIndex
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.optimization.Optimizable
import org.jblas.FloatMatrix

class CpuWeighting internal constructor(
    name: String? = null,
    numberInputRows: Int,
    minimumInputColumns: Int,
    maximumInputColumns: Int,
    numberOutputRows: Int,
    private val weights: FloatArray,
    private val accumulator: DenseAccumulator,
    private val updateRule: UpdateRule? = null) : BaseCpuContinuationLayer(name, numberInputRows, numberOutputRows, minimumInputColumns, maximumInputColumns), Optimizable {

    private val numberWeightColumns = this.numberInputRows
    private val numberWeightRows = this.numberOutputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private var blasInputMatricesOverPossibleLengths = Array(this.numberPossibleInputLengths) { index -> org.jblas.FloatMatrix(this.numberInputRows, this.possibleInputLengths[index]) }
    private var blasOutputMatricesOverPossibleLengths = Array(this.numberPossibleInputLengths) { index -> org.jblas.FloatMatrix(this.numberOutputRows, this.possibleOutputLengths[index]) }
    private var blasInputMatrix = FloatMatrix()

    private var blasWeightMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberWeightColumns)
    private val backwardWrtWeights = FloatArray(this.numberWeightEntries)

    init {
        this.blasWeightMatrix.data = this.weights
    }

    fun getWeights() =
        this.weights

    private var lengthIndex = -1

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, forwardResult: FloatArray, isTraining: Boolean) {
        this.lengthIndex = computeLengthIndex(numberInputColumns, this.minimumColumns)

        this.blasInputMatrix = this.blasInputMatricesOverPossibleLengths[this.lengthIndex]
        this.blasInputMatrix.data = input

        val blasOutputMatrix = this.blasOutputMatricesOverPossibleLengths[this.lengthIndex]
        blasOutputMatrix.data = forwardResult

        multiply(this.blasWeightMatrix, blasInputMatrix, blasOutputMatrix)
    }

    override fun computeBackwardResult(withinBatch: Int, numberInputColumns: Int, numberOutputColumns : Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardProjectionWrtInput(
            this.numberInputRows,
            numberInputColumns,
            this.weights,
            this.numberWeightRows,
            chain,
            this.numberOutputRows,
            backwardResult)

        backwardProjectionWrtWeights(
            this.numberWeightRows,
            this.numberWeightColumns,
            this.blasInputMatrix.data,
            this.numberInputRows,
            chain,
            this.numberOutputRows,
            numberOutputColumns,
            this.backwardWrtWeights)

        this.accumulator.accumulate(this.backwardWrtWeights)
    }

    override fun optimize(batchSize : Int) {
        if (this.updateRule != null) {
            updateDensely(this.weights, this.accumulator.getAccumulation(), this.numberWeightEntries, batchSize, this.updateRule)

            this.accumulator.reset()
        }
    }

}