package com.komputation.loss

import com.komputation.cpu.loss.CpuSquaredLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberRows: Int, private val maximumLength: Int, private val hasFixedLength : Boolean) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    private val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

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

fun squaredLoss(numberCategories: Int, length: Int = 1, hasFixedLength: Boolean = true) =
    SquaredLoss(numberCategories, length, hasFixedLength)