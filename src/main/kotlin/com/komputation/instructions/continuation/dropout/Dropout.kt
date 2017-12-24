package com.komputation.instructions.continuation.dropout

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.dropout.CpuDropout
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.dropout.CudaDropout
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle
import java.util.*

class Dropout internal constructor(
    private val name : String?,
    private val random : Random,
    private val keepProbability : Float) : BaseEntrywiseInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    override fun buildForCpu() =
        CpuDropout(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns, this.random, this.keepProbability)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaDropout(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            this.random,
            this.keepProbability,
            { context.createKernel(ForwardKernels.dropoutTraining()) },
            { context.createKernel(ForwardKernels.dropoutRuntime()) },
            { context.createKernel(ForwardKernels.backwardDropout()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun dropout(random: Random, keepProbability: Float) =
    dropout(null, random, keepProbability)

fun dropout(name: String?, random: Random, keepProbability: Float) =
    Dropout(name, random, keepProbability)