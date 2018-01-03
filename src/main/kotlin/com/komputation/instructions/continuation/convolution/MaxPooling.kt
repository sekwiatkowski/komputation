package com.komputation.instructions.continuation.convolution

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.maxpooling.CpuMaxPooling
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ContinuationKernels
import com.komputation.cuda.layers.continuation.maxpooling.CudaMaxPooling
import jcuda.jcublas.cublasHandle

class MaxPooling internal constructor (private val name : String?) : CpuContinuationInstruction, CudaContinuationInstruction {

    private var numberInputRows = -1
    protected var maximumNumberInputColumns = -1
    protected var minimumNumberInputColumns = -1

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.minimumNumberInputColumns = minimumNumberInputColumns
        this.maximumNumberInputColumns = maximumNumberInputColumns
    }

    override val numberOutputRows
        get() = this.numberInputRows
    override val maximumNumberOutputColumns
        get() = 1
    override val minimumNumberOutputColumns
        get() = 1

    override fun buildForCpu() =
        CpuMaxPooling(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaMaxPooling(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(ContinuationKernels.maxPooling()) },
            { context.createKernel(ContinuationKernels.backwardMaxPooling()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun maxPooling(name : String? = null) =
    MaxPooling(name)