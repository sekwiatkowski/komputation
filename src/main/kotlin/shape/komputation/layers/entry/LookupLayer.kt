package shape.komputation.layers.entry

import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.SparseAccumulator
import shape.komputation.optimization.UpdateRule
import shape.komputation.optimization.updateSparsely

class LookupLayer internal constructor(
    name : String?,
    private val vectors: Array<DoubleArray>,
    private val dimension : Int,
    maximumBatchSize: Int,
    maximumLength : Int,
    private val update: UpdateRule? = null) : EntryPoint(name), Optimizable {

    private val numberVectors = this.vectors.size

    private var input : IntArray? = null

    private val gradientAccumulator = SparseAccumulator(this.vectors.size, maximumBatchSize, maximumLength, this.dimension)

    override fun forward(input: Matrix) : DoubleMatrix {

        input as IntMatrix

        val inputEntries = input.entries
        val numberRows = input.numberRows

        this.input = inputEntries

        val result = DoubleArray(inputEntries.size * this.dimension)

        for (indexRow in 0..numberRows - 1) {

            val id = inputEntries[indexRow]

            val instance = this.vectors[id]

            for (indexColumn in 0..this.dimension - 1) {

                result[indexRow + indexColumn * numberRows] = instance[indexColumn]

            }

        }

        return DoubleMatrix(numberRows, this.dimension, result)

    }


    override fun backward(chain : DoubleMatrix): DoubleMatrix {

        this.gradientAccumulator.accumulate(this.input!!, chain.entries)

        return chain

    }

    override fun optimize(scalingFactor : Double) {

        if (this.update != null) {

            val gradientAccumulator = this.gradientAccumulator

            val size = gradientAccumulator.getSize()
            val ids = gradientAccumulator.getIds()
            val counts = gradientAccumulator.getCounts()
            val gradients = gradientAccumulator.getSums()

            updateSparsely(this.vectors, this.numberVectors, this.dimension, size, ids, counts, gradients, update)

        }

        this.gradientAccumulator.reset()

    }

}

fun lookupLayer(
    vectors: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength : Int,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    return lookupLayer(null, vectors, dimension, maximumBatchSize, maximumLength, optimizationStrategy)
}

fun lookupLayer(
    name : String? = null,
    vectors: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength: Int,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null): LookupLayer {

    val updateRule = if (optimizationStrategy != null) {

        optimizationStrategy(vectors.size, vectors[0].size)

    }
    else {

        null
    }

    return LookupLayer(name, vectors, dimension, maximumBatchSize, maximumLength, updateRule)

}