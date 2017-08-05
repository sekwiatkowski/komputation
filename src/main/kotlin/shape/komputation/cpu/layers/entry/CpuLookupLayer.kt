package shape.komputation.cpu.layers.entry

import shape.komputation.cpu.functions.lookup
import shape.komputation.cpu.layers.BaseCpuEntryPoint
import shape.komputation.cpu.optimization.SparseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateSparsely
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable
import java.util.*

class CpuLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<FloatArray>,
    private val length : Int,
    private val dimension : Int,
    private val gradientAccumulator: SparseAccumulator,
    private val update: UpdateRule? = null) : BaseCpuEntryPoint(name), Optimizable {

    private var inputEntries = IntArray(this.length)
    private var inputLength = -1
    private val forwardEntries = FloatArray(this.length * this.dimension)

    private val padding = Float.NaN

    override fun forward(input: Matrix) : FloatMatrix {

        input as IntMatrix
        this.inputEntries = input.entries
        this.inputLength = input.numberRows

        lookup(this.vectors, this.length, this.dimension, this.padding, this.inputEntries, this.forwardEntries)

        return FloatMatrix(this.dimension, this.length, this.forwardEntries)

    }


    override fun backward(chain : FloatMatrix): FloatMatrix {

        this.gradientAccumulator.accumulate(this.inputEntries, this.inputLength-1, chain.entries)

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