package shape.komputation.layers.entry

import shape.komputation.layers.BaseEntryPoint
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.*

class CpuLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<DoubleArray>,
    private val dimension : Int,
    private val gradientAccumulator: SparseAccumulator,
    private val update: UpdateRule? = null) : BaseEntryPoint(name), Optimizable {

    private var input : IntArray? = null

    override fun forward(input: Matrix) : DoubleMatrix {

        input as IntMatrix

        val inputEntries = input.entries
        val inputSize = inputEntries.size

        this.input = inputEntries

        /*
            word^(1)_1   word^(2)_1   ...   word^(T)_1
            word^(1)_2   word^(2)_2   ...   word^(T)_2
            ...          ...                ....
            word^(1)_d   word^(2)_d   ...   word^(T)_d
        */


        val result = DoubleArray(inputSize * this.dimension)

        var start = 0

        for (indexInput in 0..inputSize - 1) {

            val id = inputEntries[indexInput]

            val vector = this.vectors[id]

            for (indexDimension in 0..this.dimension - 1) {

                result[start++] = vector[indexDimension]

            }

        }

        return DoubleMatrix(this.dimension, inputSize, result)

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

            updateSparsely(this.vectors, this.dimension, size, ids, counts, gradients, update)

        }

        this.gradientAccumulator.reset()

    }

}

class LookupLayer(
    private val name : String? = null,
    private val vectors: Array<DoubleArray>,
    private val dimension : Int,
    private val maximumBatchSize : Int,
    private val maximumLength: Int,
    private val optimizationStrategy : OptimizationStrategy?) : CpuEntryPointInstruction {

    override fun buildForCpu(): CpuLookupLayer {

        val updateRule = if (this.optimizationStrategy != null) {

            this.optimizationStrategy.invoke(this.vectors.size, this.vectors[0].size)

        }
        else {

            null
        }

        val sparseAccumulator = SparseAccumulator(this.vectors.size, this.maximumBatchSize, this.maximumLength, this.dimension)

        return CpuLookupLayer(name, vectors, dimension, sparseAccumulator, updateRule)

    }


}

fun lookupLayer(
    vectors: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength : Int,
    optimizationStrategy : ((numberRows : Int, numberColumns : Int) -> UpdateRule)? = null) =

    lookupLayer(null, vectors, dimension, maximumBatchSize, maximumLength, optimizationStrategy)

fun lookupLayer(
    name : String? = null,
    vectors: Array<DoubleArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength: Int,
    optimizationStrategy : OptimizationStrategy? = null) =

    LookupLayer(
        name,
        vectors,
        dimension,
        maximumBatchSize,
        maximumLength,
        optimizationStrategy
    )