package com.komputation.loss

import com.komputation.cpu.loss.CpuLogisticLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaLogisticLoss

class LogisticLoss(private val numberColumns: Int, private val hasFixedLength : Boolean) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuLogisticLoss(this.numberColumns, this.hasFixedLength)

    override fun buildForCuda(context: CudaContext) =
        CudaLogisticLoss(
            this.numberColumns,
            { context.createKernel(LossKernels.logisticLoss()) },
            { context.createKernel(LossKernels.backwardLogisticLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun logisticLoss(length: Int = 1, hasFixedLength: Boolean = true) =
    LogisticLoss(length, hasFixedLength)