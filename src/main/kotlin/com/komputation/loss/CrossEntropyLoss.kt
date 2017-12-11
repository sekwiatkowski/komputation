package com.komputation.loss

import com.komputation.cpu.loss.CpuCrossEntropyLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaCrossEntropyLoss

class CrossEntropyLoss(private val numberRows: Int, private val maximumLength: Int, private val hasFixedLength : Boolean) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    private val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

    override fun buildForCpu() =
        CpuCrossEntropyLoss(this.numberRows, this.minimumLength, this.maximumLength)

    override fun buildForCuda(context: CudaContext) =
        CudaCrossEntropyLoss(
            this.numberRows,
            this.maximumLength,
            { context.createKernel(LossKernels.crossEntropyLoss()) },
            { context.createKernel(LossKernels.backwardCrossEntropyLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun crossEntropyLoss(numberCategories: Int, length: Int = 1, hasFixedLength: Boolean = true) =
    CrossEntropyLoss(numberCategories, length, hasFixedLength)