package com.komputation.instructions.loss

import com.komputation.cpu.instructions.CpuLossFunctionInstruction
import com.komputation.cpu.loss.CpuLogisticLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaLossFunctionInstruction
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaLogisticLoss
import com.komputation.instructions.BaseLossInstruction

class LogisticLoss(name : String?) : BaseLossInstruction(name), CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuLogisticLoss(this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext) =
        CudaLogisticLoss(
            this.maximumNumberInputColumns,
            { context.createKernel(LossKernels.logisticLoss()) },
            { context.createKernel(LossKernels.backwardLogisticLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun logisticLoss(name : String? = null) =
    LogisticLoss(name)