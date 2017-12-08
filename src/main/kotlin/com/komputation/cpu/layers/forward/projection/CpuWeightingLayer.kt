package com.komputation.cpu.layers.forward.projection

import com.komputation.cpu.functions.backwardProjectionWrtInput
import com.komputation.cpu.functions.backwardProjectionWrtWeights
import com.komputation.cpu.functions.multiply
import com.komputation.cpu.layers.BaseCpuVariableLengthForwardLayer
import com.komputation.cpu.optimization.DenseAccumulator
import com.komputation.cpu.optimization.UpdateRule
import com.komputation.cpu.optimization.updateDensely
import com.komputation.layers.Resourceful
import com.komputation.optimization.Optimizable

class CpuWeightingLayer internal constructor(
    name : String? = null,
    private val weights : FloatArray,
    numberInputRows : Int,
    minimumInputColumns: Int,
    maximumInputColumns: Int,
    numberOutputRows: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null) : BaseCpuVariableLengthForwardLayer(name, numberInputRows, numberOutputRows, minimumInputColumns, maximumInputColumns), Resourceful, Optimizable {

    private val numberWeightColumns = this.numberInputRows
    private val numberWeightRows = this.numberOutputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private var blasInputMatricesOverPossibleLengths = emptyArray<org.jblas.FloatMatrix>()
    private var blasOutputMatricesOverPossibleLengths = emptyArray<org.jblas.FloatMatrix>()

    private var blasWeightMatrix = org.jblas.FloatMatrix()
    private val backwardWrtWeights = FloatArray(this.numberWeightEntries)

    fun getWeights() =
        this.weights

    override fun acquire(maximumBatchSize: Int) {
        super.acquire(maximumBatchSize)

        this.blasOutputMatricesOverPossibleLengths = Array(this.numberPossibleLengths) { index -> org.jblas.FloatMatrix(this.numberOutputRows, this.possibleLengths[index]) }
        this.blasInputMatricesOverPossibleLengths = Array(this.numberPossibleLengths) { index -> org.jblas.FloatMatrix(this.numberInputRows, this.possibleLengths[index]) }
        this.blasWeightMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberWeightColumns)
        this.blasWeightMatrix.data = this.weights
    }

    override fun computeNumberOutputColumns(lengthIndex: Int, length: Int) =
        length

    override fun computeForwardResult(withinBatch: Int, numberInputColumns: Int, input: FloatArray, isTraining: Boolean, forwardResult: FloatArray) {
        val blasInputMatrix = this.blasInputMatricesOverPossibleLengths[this.lengthIndex]
        blasInputMatrix.data = input

        val blasOutputMatrix = this.blasOutputMatricesOverPossibleLengths[this.lengthIndex]
        blasOutputMatrix.data = forwardResult

        multiply(this.blasWeightMatrix, blasInputMatrix, blasOutputMatrix)
    }

    override fun computeBackwardResult(withinBatch: Int, forwardResult : FloatArray, chain: FloatArray, backwardResult: FloatArray) {
        backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            this.weights,
            this.numberWeightRows,
            chain,
            this.numberOutputRows,
            this.backwardResult)

        val blasInputMatrix = this.blasInputMatricesOverPossibleLengths[this.lengthIndex]
        backwardProjectionWrtWeights(
            this.numberWeightRows,
            this.numberWeightColumns,
            blasInputMatrix.data,
            this.numberInputRows,
            chain,
            this.numberOutputRows,
            this.numberOutputColumns,
            this.backwardWrtWeights)

        this.weightAccumulator.accumulate(this.backwardWrtWeights)
    }

    override fun optimize(batchSize : Int) {
        if (this.weightUpdateRule != null) {
            updateDensely(this.weights, this.weightAccumulator.getAccumulation(), this.numberWeightEntries, batchSize, this.weightUpdateRule)

            this.weightAccumulator.reset()
        }
    }

}