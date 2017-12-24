package com.komputation.cuda.instructions

import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.instructions.ContinuationInstruction
import jcuda.jcublas.cublasHandle

interface CudaContinuationInstruction : ContinuationInstruction {
    fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle) : CudaContinuation
}