package shape.komputation.cpu.layers.forward.projection

import shape.komputation.cpu.functions.addBias
import shape.komputation.cpu.functions.backwardProjectionWrtBias
import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateDensely
import shape.komputation.matrix.FloatMatrix
import shape.komputation.optimization.Optimizable

class CpuBiasLayer internal constructor(
    name : String? = null,
    private val bias : FloatArray,
    private val biasAccumulator: DenseAccumulator,
    private val biasUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Optimizable {

    private val numberBiasEntries = bias.size
    private val backwardResult = FloatArray(this.numberBiasEntries)

    private var numberInputRows = -1
    private var numberInputColumns = -1
    private var numberInputEntries = -1

    override fun forward(input: FloatMatrix, isTraining : Boolean) : FloatMatrix {

        this.numberInputRows = input.numberRows
        this.numberInputColumns = input.numberColumns
        this.numberInputEntries = numberInputRows * numberInputColumns

        val result = FloatArray(this.numberInputEntries)

        addBias(input.entries, this.numberInputRows, this.numberInputEntries, this.bias, result)

        return FloatMatrix(this.numberInputRows, this.numberInputColumns, result)

    }

    override fun backward(chain : FloatMatrix) : FloatMatrix {

        val chainEntries = chain.entries
        val numberChainRows = chain.numberRows
        val numberChainColumns = chain.numberColumns

        backwardProjectionWrtBias(this.numberBiasEntries, chainEntries, numberChainRows, numberChainColumns, this.backwardResult)

        this.biasAccumulator.accumulate(this.backwardResult)

        return chain

    }

    override fun optimize(scalingFactor : Float) {

        if (this.biasUpdateRule != null) {

            updateDensely(this.bias, this.biasAccumulator.getAccumulation(), this.numberBiasEntries, scalingFactor, this.biasUpdateRule)

            this.biasAccumulator.reset()

        }

    }

}