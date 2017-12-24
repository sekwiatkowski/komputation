package com.komputation.cuda.instructions

import jcuda.jcublas.cublasHandle
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.continuation.CudaActivation

interface CudaActivationInstruction : CudaContinuationInstruction {
    override fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle): CudaActivation
}