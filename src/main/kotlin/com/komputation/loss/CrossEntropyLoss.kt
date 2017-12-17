package com.komputation.loss

import com.komputation.cpu.loss.CpuCrossEntropyLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaCrossEntropyLoss

class CrossEntropyLoss(private val numberRows: Int, private val minimumLength: Int, private val maximumLength: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

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

fun crossEntropyLoss(numberCategories: Int, minimumLength : Int = 1, maximumLength: Int = minimumLength) =
    CrossEntropyLoss(numberCategories, minimumLength, maximumLength)