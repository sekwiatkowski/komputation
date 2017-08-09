package shape.komputation.cpu.layers.entry

import shape.komputation.cpu.functions.lookup
import shape.komputation.cpu.layers.BaseCpuEntryPoint
import shape.komputation.cpu.optimization.SparseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cpu.optimization.updateSparsely
import shape.komputation.matrix.IntMatrix
import shape.komputation.matrix.Matrix
import shape.komputation.optimization.Optimizable

class CpuLookupLayer internal constructor(
    name : String?,
    private val vectors: Array<FloatArray>,
    private val minimumLength: Int,
    private val maximumLength: Int,
    private val dimension : Int,
    private val gradientAccumulator: SparseAccumulator,
    private val update: UpdateRule? = null) : BaseCpuEntryPoint(name), Optimizable {

    override var forwardResult = FloatArray(0)
    override val numberOutputRows = this.dimension
    override var numberOutputColumns = -1

    private var inputEntries = IntArray(0)

    private val numberLengths = this.maximumLength - this.minimumLength + 1
    private val forwardResultsOverPossibleLengths = Array(this.numberLengths) { index -> FloatArray((index+this.minimumLength)*this.dimension) }

    override fun forward(input: Matrix): FloatArray {

        input as IntMatrix

        this.inputEntries = input.entries
        this.numberOutputColumns = this.inputEntries.size

        this.forwardResult = this.forwardResultsOverPossibleLengths[this.numberOutputColumns - this.minimumLength]

        lookup(this.vectors, this.dimension, this.numberOutputColumns, this.inputEntries, this.forwardResult)

        return this.forwardResult

    }

    override fun backward(chain : FloatArray) {

        this.gradientAccumulator.accumulate(this.inputEntries, this.numberOutputColumns, chain)

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