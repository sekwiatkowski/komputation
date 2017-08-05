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
    private val numberRows : Int,
    private val numberColumns : Int,
    private val bias : FloatArray,
    private val biasAccumulator: DenseAccumulator,
    private val biasUpdateRule: UpdateRule? = null) : BaseCpuForwardLayer(name), Optimizable {

    private val numberBiasEntries = this.bias.size
    private val backwardResult = FloatArray(this.numberBiasEntries)

    private val numberEntries = this.numberRows * this.numberColumns

    private val result = FloatArray(this.numberEntries)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean) : FloatMatrix {

        addBias(input.entries, this.numberRows, this.numberColumns, this.numberEntries, this.bias, this.result)

        return FloatMatrix(this.numberRows, this.numberColumns, this.result)

    }

    override fun backward(withinBatch : Int, chain : FloatMatrix) : FloatMatrix {

        val chainEntries = chain.entries

        backwardProjectionWrtBias(this.numberBiasEntries, chainEntries, this.numberRows, this.numberColumns, this.backwardResult)

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