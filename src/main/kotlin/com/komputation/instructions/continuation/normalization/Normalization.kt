package com.komputation.instructions.continuation.normalization

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.normalization.CpuNormalization
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ContinuationKernels
import com.komputation.cuda.layers.continuation.normalization.CudaNormalization
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class NormalizationLayer internal constructor(private val name : String?) : BaseEntrywiseInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    override fun buildForCpu() =
        CpuNormalization(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaNormalization(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ContinuationKernels.normalization()) },
            { context.createKernel(ContinuationKernels.backwardNormalization()) },
            context.maximumNumberOfThreadsPerBlock,
            context.warpSize)

}

fun normalization(name : String?) =
    NormalizationLayer(name)