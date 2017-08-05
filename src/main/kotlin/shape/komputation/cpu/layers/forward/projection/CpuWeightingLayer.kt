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
import java.util.*

class CpuWeightingLayer internal constructor(
    name : String? = null,
    private val weights : FloatArray,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val numberWeightRows: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Optimizable {

    private val numberInputEntries = this.numberInputRows * this.numberInputColumns

    private val numberWeightColumns = this.numberInputRows
    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val numberChainRows = this.numberWeightRows
    private val numberChainColumns = this.numberInputColumns

    private val backwardWrtInput = FloatArray(this.numberInputEntries)
    private val backwardWrtWeights = FloatArray(this.numberWeightEntries)

    private val blasWeightMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberWeightColumns)
    private val blasInputMatrix = org.jblas.FloatMatrix(this.numberInputRows, this.numberInputColumns)
    private val blasResultMatrix = org.jblas.FloatMatrix(this.numberWeightRows, this.numberInputColumns)

    init {

        this.blasWeightMatrix.data = weights

    }

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.blasInputMatrix.data = input.entries

        multiply(this.blasWeightMatrix, this.blasInputMatrix, this.blasResultMatrix)

        return FloatMatrix(this.numberWeightRows, this.numberInputColumns, this.blasResultMatrix.data)

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        val chainEntries = chain.entries

        backwardProjectionWrtInput(
            this.numberInputRows,
            this.numberInputColumns,
            this.weights,
            this.numberWeightRows,
            chainEntries,
            this.numberChainRows,
            this.backwardWrtInput)

        backwardProjectionWrtWeights(
            this.numberWeightRows,
            this.numberWeightColumns,
            this.blasInputMatrix.data,
            this.numberInputRows,
            chainEntries,
            this.numberChainRows,
            this.numberChainColumns,
            this.backwardWrtWeights)

        this.weightAccumulator.accumulate(this.backwardWrtWeights)

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, this.backwardWrtInput)

    }

    override fun optimize(scalingFactor : Float) {

        if (this.weightUpdateRule != null) {

            updateDensely(this.weights, this.weightAccumulator.getAccumulation(), this.numberWeightEntries, scalingFactor, this.weightUpdateRule)

            this.weightAccumulator.reset()

        }

    }

}