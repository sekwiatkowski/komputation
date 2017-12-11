package com.komputation.loss

import com.komputation.cpu.loss.CpuLogisticLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaLogisticLoss

class LogisticLoss(private val maximumLength: Int, private val hasFixedLength : Boolean) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    private val minimumLength = if(this.hasFixedLength) this.maximumLength else 1

    override fun buildForCpu() =
        CpuLogisticLoss(this.minimumLength, this.maximumLength)

    override fun buildForCuda(context: CudaContext) =
        CudaLogisticLoss(
            this.maximumLength,
            { context.createKernel(LossKernels.logisticLoss()) },
            { context.createKernel(LossKernels.backwardLogisticLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun logisticLoss(length: Int = 1, hasFixedLength: Boolean = true) =
    LogisticLoss(length, hasFixedLength)