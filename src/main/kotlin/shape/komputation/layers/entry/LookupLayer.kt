package shape.komputation.layers.entry

import shape.komputation.cpu.layers.entry.CpuLookupLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.EntryKernels
import shape.komputation.cuda.layers.entry.CudaLookupLayer
import shape.komputation.layers.CpuEntryPointInstruction
import shape.komputation.layers.CudaEntryPointInstruction
import shape.komputation.optimization.OptimizationInstruction

class LookupLayer(
    private val name : String? = null,
    private val vectors: Array<FloatArray>,
    private val maximumLength: Int,
    private val hasFixedLength: Boolean,
    private val dimension : Int,
    private val optimization : OptimizationInstruction?) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override fun buildForCpu(): CpuLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCpu().invoke(this.vectors.size, this.dimension)

        }
        else {

            null
        }

        val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

        return CpuLookupLayer(this.name, this.vectors, minimumLength, this.maximumLength, this.dimension, updateRule)

    }

    override fun buildForCuda(context: CudaContext): CudaLookupLayer {

        val updateRule = if (this.optimization != null) {

            this.optimization.buildForCuda(context).invoke(this.vectors.size, this.dimension, this.maximumLength)

        }
        else {

            null
        }

        return CudaLookupLayer(
            this.name,
            this.vectors,
            this.maximumLength,
            this.dimension,
            updateRule,
            { context.createKernel(EntryKernels.lookup())},
            context.maximumNumberOfThreadsPerBlock)

    }


}

fun lookupLayer(
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    lookupLayer(
        null,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        optimization
    )

fun lookupLayer(
    name: String? = null,
    vectors: Array<FloatArray>,
    maximumLength: Int,
    hasFixedLength: Boolean,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    LookupLayer(
        name,
        vectors,
        maximumLength,
        hasFixedLength,
        dimension,
        optimization
    )