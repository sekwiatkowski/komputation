package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuTanh
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.activation.CudaTanh
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class Tanh internal constructor(private val name : String?) : BaseEntrywiseInstruction(), ActivationInstruction {

    override fun buildForCpu() =
        CpuTanh(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaTanh(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ForwardKernels.tanh()) },
            { context.createKernel(ForwardKernels.backwardTanh()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun tanh(name : String? = null) =
    Tanh(name)