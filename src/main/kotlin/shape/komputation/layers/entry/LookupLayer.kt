package shape.komputation.layers.entry

import shape.komputation.cpu.entry.CpuLookupLayer
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.SparseAccumulator
import shape.komputation.optimization.UpdateRule

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

        return CpuLookupLayer(this.name, this.vectors, this.dimension, sparseAccumulator, updateRule)

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