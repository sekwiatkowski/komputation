package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuLookupLayer
import shape.komputation.cpu.optimization.SparseAccumulator
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.optimization.OptimizationInstruction

class LookupLayer(
    private val name : String? = null,
    private val vectors: Array<FloatArray>,
    private val dimension : Int,
    private val maximumBatchSize : Int,
    private val maximumLength: Int,
    private val optimization : OptimizationInstruction?) : CpuEntryPointInstruction {

    override fun buildForCpu(): CpuLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCpu().invoke(this.vectors.size, this.vectors[0].size)

        }
        else {

            null
        }

        val sparseAccumulator = SparseAccumulator(this.vectors.size, this.maximumBatchSize, this.maximumLength, this.dimension)

        return CpuLookupLayer(this.name, this.vectors, this.dimension, sparseAccumulator, updateRule)

    }


}

fun lookupLayer(
    vectors: Array<FloatArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength : Int,
    optimization: OptimizationInstruction? = null) =

    lookupLayer(null, vectors, dimension, maximumBatchSize, maximumLength, optimization)

fun lookupLayer(
    name : String? = null,
    vectors: Array<FloatArray>,
    dimension : Int,
    maximumBatchSize : Int,
    maximumLength: Int,
    optimization: OptimizationInstruction? = null) =

    LookupLayer(
        name,
        vectors,
        dimension,
        maximumBatchSize,
        maximumLength,
        optimization
    )