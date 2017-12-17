package com.komputation.loss

import com.komputation.cpu.loss.CpuSquaredLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberRows: Int, private val minimumLength: Int, private val maximumLength: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuSquaredLoss(this.numberRows, this.minimumLength, this.maximumLength)

    override fun buildForCuda(context: CudaContext) =
        CudaSquaredLoss(
            this.numberRows,
            this.maximumLength,
            { context.createKernel(LossKernels.squaredLoss()) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun squaredLoss(numberCategories: Int, minimumLength: Int = 1, maximumLength: Int = minimumLength) =
    SquaredLoss(numberCategories, minimumLength, maximumLength)