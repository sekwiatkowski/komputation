package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuRelu
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ContinuationKernels
import com.komputation.cuda.layers.continuation.activation.CudaRelu
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class Relu internal constructor(private val name : String?) : BaseEntrywiseInstruction(), ActivationInstruction {

    override fun buildForCpu() =
        CpuRelu(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) =
        CudaRelu(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ContinuationKernels.relu()) },
            { context.createKernel(ContinuationKernels.backwardRelu()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun relu(name : String? = null) =
    Relu(name)