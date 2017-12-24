package com.komputation.instructions.loss

import com.komputation.cpu.instructions.CpuLossFunctionInstruction
import com.komputation.cpu.loss.CpuSquaredLoss
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaLossFunctionInstruction
import com.komputation.cuda.kernels.LossKernels
import com.komputation.cuda.loss.CudaSquaredLoss
import com.komputation.instructions.BaseLossInstruction

class SquaredLoss(name : String?) : BaseLossInstruction(name), CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =
        CpuSquaredLoss(this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext) =
        CudaSquaredLoss(
            this.numberInputRows,
            this.maximumNumberInputColumns,
            { context.createKernel(LossKernels.squaredLoss()) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun squaredLoss(name : String? = null) =
    SquaredLoss(name)