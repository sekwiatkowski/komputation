package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuLookupLayer
import shape.komputation.cpu.optimization.SparseAccumulator
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.optimization.OptimizationInstruction

class LookupLayer(
    private val name : String? = null,
    private val vectors: Array<FloatArray>,
    private val maximumLength: Int,
    private val hasFixedLength: Boolean,
    private val dimension : Int,
    private val maximumBatchSize : Int,
    private val optimization : OptimizationInstruction?) : CpuEntryPointInstruction {

    override fun buildForCpu(): CpuLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCpu().invoke(this.vectors.size, this.vectors[0].size)

        }
        else {

            null
        }

        val sparseAccumulator = SparseAccumulator(this.vectors.size, this.maximumBatchSize, this.maximumLength, this.dimension)

        val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

        return CpuLookupLayer(this.name, this.vectors, minimumLength, this.maximumLength, this.dimension, sparseAccumulator, updateRule)

    }


}


fun lookupLayer(
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    maximumBatchSize: Int,
    optimization: OptimizationInstruction? = null) =

    lookupLayer(
        null,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        maximumBatchSize,
        optimization
    )

fun lookupLayer(
    name: String? = null,
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    maximumBatchSize: Int,
    optimization: OptimizationInstruction? = null) =

    LookupLayer(
        name,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        maximumBatchSize,
        optimization
    )