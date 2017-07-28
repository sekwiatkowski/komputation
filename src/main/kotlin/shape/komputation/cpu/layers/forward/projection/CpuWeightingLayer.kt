package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.functions.backwardProjectionWrtInput
import shape.komputation.cpu.functions.backwardProjectionWrtWeights
import shape.komputation.cpu.functions.multiply
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuWeightingLayer internal constructor(
    name : String? = null,
    private val weights : FloatArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Optimizable {

    private var inputEntries = FloatArray(0)
    private var numberInputRows = -1
    private var numberInputColumns = -1

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns
    private val backwardWrtWeights = FloatArray(this.numberWeightEntries)

    private val blasWeightMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberWeightColumns)

    init {

        blasWeightMatrix.data = weights

    }

    override fun forward(input: FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns
        val numberInputEntries = this.numberInputRows * this.numberInputColumns

        this.inputEntries = FloatArray(numberInputEntries)
        System.arraycopy(input.entries, 0, this.inputEntries, 0, numberInputEntries)

        val blasInputMatrix = org.jblas.FloatMatrix(this.numberInputRows, this.numberInputColumns)
        blasInputMatrix.data = this.inputEntries

        val blasResultMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberInputColumns)
        multiply(this.blasWeightMatrix, blasInputMatrix, blasResultMatrix)

        return FloatMatrix(this.numberWeightRows, this.numberInputColumns, blasResultMatrix.data)

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        val numberInputEntries = this.numberInputRows * this.numberInputColumns

        val backwardWrtInput = backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            numberInputEntries,
            this.weights,
            this.numberWeightRows,
            chainEntries,
            numberChainRows)

        backwardProjectionWrtWeights(
            this.numberWeightRows,
            this.numberWeightColumns,
            this.inputEntries,
            this.numberInputRows,
            chainEntries,
            numberChainRows,
            numberChainColumns,
            this.backwardWrtWeights)

        this.weightAccumulator.accumulate(this.backwardWrtWeights)

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, backwardWrtInput)

    }

    override fun optimize(scalingFactor : Float) {

        if (this.weightUpdateRule != null) {

            updateDensely(this.weights, this.weightAccumulator.getAccumulation(), this.numberWeightEntries, scalingFactor, this.weightUpdateRule)

            this.weightAccumulator.reset()

        }

    }

}