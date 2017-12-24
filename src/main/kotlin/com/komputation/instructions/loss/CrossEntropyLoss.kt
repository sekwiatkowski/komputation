package com.komputation.instructions.loss

import com.komputation.cpu.instructions.CpuLossFunctionInstruction
import com.komputation.cpu.loss.CpuCrossEntropyLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaLossFunctionInstruction
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaCrossEntropyLoss
import com.komputation.instructions.BaseLossInstruction

class CrossEntropyLoss(name : String?) : BaseLossInstruction(name), CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuCrossEntropyLoss(this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext) =
        CudaCrossEntropyLoss(
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(LossKernels.crossEntropyLoss()) },
            { context.createKernel(LossKernels.backwardCrossEntropyLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun crossEntropyLoss(name : String? = null) =
    CrossEntropyLoss(name)