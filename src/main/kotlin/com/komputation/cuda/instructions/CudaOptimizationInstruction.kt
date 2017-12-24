package com.komputation.cuda.instructions

import com.komputation.cuda.CudaContext
import com.komputation.cuda.optimization.CudaOptimizationStrategy

interface CudaOptimizationInstruction {

    fun buildForCuda(context: CudaContext) : CudaOptimizationStrategy

}