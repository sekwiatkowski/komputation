package com.komputation.loss

import com.komputation.cpu.loss.CpuSquaredLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberRows: Int, private val numberColumns: Int, private val hasFixedLength : Boolean) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuSquaredLoss(this.numberRows, this.numberColumns, this.hasFixedLength)

    override fun buildForCuda(context: CudaContext) =
        CudaSquaredLoss(
            this.numberRows,
            this.numberColumns,
            { context.createKernel(LossKernels.squaredLoss()) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun squaredLoss(numberCategories: Int, length: Int = 1, hasFixedLength: Boolean = true) =
    SquaredLoss(numberCategories, length, hasFixedLength)