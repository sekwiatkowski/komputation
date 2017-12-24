package com.komputation.cuda.instructions

import com.komputation.cuda.CudaContext
import com.komputation.cuda.loss.CudaLossFunction
import com.komputation.instructions.LossInstruction

interface CudaLossFunctionInstruction : LossInstruction {

    fun buildForCuda(context: CudaContext) : CudaLossFunction

}