package shape.komputation.cpu.layers.entry

import shape.komputation.cpu.layers.BaseCpuEntryPoint
import shape.komputation.cpu.optimization.SparseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateSparsely
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

class CpuLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<FloatArray>,
    private val dimension : Int,
    private val gradientAccumulator: SparseAccumulator,
    private val update: UpdateRule? = null) : BaseCpuEntryPoint(name), Optimizable {

    private var input : IntArray? = null

    override fun forward(input: Matrix) : FloatMatrix {

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

        val result = FloatArray(inputSize * this.dimension)

        var start = 0

        for (indexInput in 0..inputSize - 1) {

            val id = inputEntries[indexInput]

            val vector = this.vectors[id]

            for (indexDimension in 0..this.dimension - 1) {

                result[start++] = vector[indexDimension]

            }

        }

        return FloatMatrix(this.dimension, inputSize, result)

    }


    override fun backward(chain : FloatMatrix): FloatMatrix {

        this.gradientAccumulator.accumulate(this.input!!, chain.entries)

        return chain

    }

    override fun optimize(scalingFactor : Float) {

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