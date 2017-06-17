package shape.komputation.layers.entry

import shape.komputation.layers.OptimizableLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.SparseAccumulator
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateSparsely

class LookupLayer(
    name : String?,
    private val data: Array<DoubleArray>,
    private val dimension : Int,
    maximumBatchSize: Int,
    private val update: UpdateRule? = null) : EntryPoint(name), OptimizableLayer {

    private var input : IntArray? = null

    private val gradientAccumulator = if(update != null) SparseAccumulator(maximumBatchSize) else null

    override fun forward(input: Matrix) : DoubleMatrix {

        input as IntMatrix

        val inputEntries = input.entries
        val numberRows = input.numberRows

        this.input = inputEntries

        val result = DoubleArray(inputEntries.size * dimension)

        for (indexRow in 0..numberRows - 1) {

            val id = inputEntries[indexRow]

            val instance = data[id]

            for (indexColumn in 0..dimension - 1) {

                result[indexRow + indexColumn * numberRows] = instance[indexColumn]

            }

        }

        return DoubleMatrix(numberRows, dimension, result)

    }


    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        this.gradientAccumulator!!.accumulate(this.input!!, chain.entries)

        return chain

    }

    override fun optimize() {

        if (update != null) {

            val gradientAccumulator = this.gradientAccumulator!!

            val (inputs, gradients) = gradientAccumulator.getAccumulation()

            updateSparsely(data, dimension, inputs, gradients, update)

            gradientAccumulator.reset()

        }

    }

}

fun createLookupLayer(
    data: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    return createLookupLayer(null, data, dimension, maximumBatchSize, optimizationStrategy)
}

fun createLookupLayer(
    name : String? = null,
    data: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    val updateRule = if (optimizationStrategy != null) {

        optimizationStrategy(data.size, data[0].size)

    }
    else {

        null
    }

    return LookupLayer(name, data, dimension, maximumBatchSize, updateRule)

}