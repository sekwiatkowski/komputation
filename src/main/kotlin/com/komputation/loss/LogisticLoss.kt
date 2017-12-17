package com.komputation.loss

import com.komputation.cpu.loss.CpuLogisticLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaLogisticLoss

class LogisticLoss(private val minimumLength : Int, private val maximumLength: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

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

fun logisticLoss(minimumLength: Int = 1, maximumLength: Int = minimumLength) =
    LogisticLoss(minimumLength, maximumLength)