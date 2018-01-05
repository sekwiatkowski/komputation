package com.komputation.instructions.entry

import com.komputation.cpu.functions.copy
import com.komputation.cpu.instructions.CpuEntryPointInstruction
import com.komputation.cpu.layers.entry.CpuLookup
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaEntryPointInstruction
import com.komputation.cuda.kernels.ArrayKernels
import com.komputation.cuda.kernels.EntryKernels
import com.komputation.cuda.layers.entry.CudaLookup
import com.komputation.optimization.OptimizationInstruction

class Lookup(
    private val name : String? = null,
    private val vectors: Array<FloatArray>,
    private val minimumLength: Int,
    private val maximumLength: Int,
    private val dimension : Int,
    private val optimization : OptimizationInstruction?) : CpuEntryPointInstruction, CudaEntryPointInstruction {

    override val numberOutputRows
        get() = this.dimension
    override val minimumNumberOutputColumns
        get() = this.minimumLength
    override val maximumNumberOutputColumns
        get() = this.maximumLength

    override fun buildForCpu(): CpuLookup {
        val updateRule = if (this.optimization != null) {
            this.optimization.buildForCpu().invoke(this.vectors.size, this.dimension)
        }
        else {
            null
        }
        return CpuLookup(this.name, this.vectors, this.dimension, this.minimumNumberOutputColumns, this.maximumLength, updateRule)
    }

    override fun buildForCuda(context: CudaContext): CudaLookup {
        val updateRule = if (this.optimization != null) {
            this.optimization.buildForCuda(context).invoke(this.vectors.size, this.dimension, 1)
        }
        else {
            null
        }

        val concatenation = FloatArray(this.vectors.size * this.dimension)
        for ((index, vector) in this.vectors.withIndex()) {
            copy(vector, index * this.dimension, this.dimension, concatenation)
        }

        return CudaLookup(
            this.name,
            concatenation,
            this.maximumLength,
            this.minimumLength == this.maximumLength,
            this.dimension,
            updateRule,
            { context.createKernel(EntryKernels.lookup())},
            { context.createKernel(ArrayKernels.groupSum())},
            context.maximumNumberOfThreadsPerBlock)
    }

}

fun lookup(
    vectors: Array<FloatArray>,
    maximumLength: Int,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    lookup(
        null,
        vectors,
        1,
        maximumLength,
        dimension,
        optimization
    )

fun lookup(
    vectors: Array<FloatArray>,
    minimumLength: Int,
    maximumLength: Int,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    lookup(
        null,
        vectors,
        minimumLength,
        maximumLength,
        dimension,
        optimization
    )

fun lookup(
    name: String? = null,
    vectors: Array<FloatArray>,
    maximumLength: Int,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    lookup(
        name,
        vectors,
        1,
        maximumLength,
        dimension,
        optimization
    )

fun lookup(
    name: String? = null,
    vectors: Array<FloatArray>,
    minimumLength: Int,
    maximumLength: Int,
    dimension: Int,
    optimization: OptimizationInstruction? = null) =

    Lookup(
        name,
        vectors,
        minimumLength,
        maximumLength,
        dimension,
        optimization
    )