package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuSigmoid
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.activation.CudaSigmoid
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class Sigmoid internal constructor(
    private val name : String?) : BaseEntrywiseInstruction(), ActivationInstruction {

    override fun buildForCpu() =
        CpuSigmoid(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) =
        CudaSigmoid(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ForwardKernels.sigmoid()) },
            { context.createKernel(ForwardKernels.backwardSigmoid()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun sigmoid(name : String? = null) =
    Sigmoid(name)