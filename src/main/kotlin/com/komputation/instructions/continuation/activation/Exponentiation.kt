package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuExponentiation
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.activation.CudaExponentiation
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class Exponentiation internal constructor(private val name : String?) : BaseEntrywiseInstruction(), ActivationInstruction {

    override fun buildForCpu() =
        CpuExponentiation(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaExponentiation(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ForwardKernels.exponentiation()) },
            { context.createKernel(ForwardKernels.backwardExponentiation()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun exponentiation(name : String?) =
    Exponentiation(name)