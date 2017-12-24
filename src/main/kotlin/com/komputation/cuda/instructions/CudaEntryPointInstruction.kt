package com.komputation.cuda.instructions

import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.CudaEntryPoint
import com.komputation.instructions.EntryPointInstruction
import com.komputation.instructions.Instruction

interface CudaEntryPointInstruction : EntryPointInstruction {

    fun buildForCuda(context : CudaContext) : CudaEntryPoint

}