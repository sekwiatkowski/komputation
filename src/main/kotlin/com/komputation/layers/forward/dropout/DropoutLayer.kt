package com.komputation.layers.forward.dropout

import com.komputation.cpu.layers.forward.dropout.CpuDropoutLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.forward.dropout.CudaDropoutLayer
import com.komputation.layers.CpuForwardLayerInstruction
import com.komputation.layers.CudaForwardLayerInstruction
import jcuda.jcublas.cublasHandle
import java.util.*

class DropoutLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val maximumColumns : Int,
    private val hasFixedLength : Boolean,
    private val random : Random,
    private val keepProbability : Float) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =
        CpuDropoutLayer(this.name, this.numberRows, if(this.hasFixedLength) this.maximumColumns else 1, this.maximumColumns, this.random, this.keepProbability)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaDropoutLayer(
            this.name,
            this.numberRows,
            this.maximumColumns,
            this.random,
            this.keepProbability,
            { context.createKernel(ForwardKernels.dropoutTraining()) },
            { context.createKernel(ForwardKernels.dropoutRuntime()) },
            { context.createKernel(ForwardKernels.backwardDropout()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun dropoutLayer(numberRows: Int, random: Random, keepProbability: Float) =
    dropoutLayer(null, numberRows, random, keepProbability)

fun dropoutLayer(name: String?, numberRows: Int, random: Random, keepProbability: Float) =
    dropoutLayer(name, numberRows, 1, true, random, keepProbability)

fun dropoutLayer(numberRows: Int, numberColumns: Int, hasFixedLength: Boolean, random: Random, keepProbability: Float) =
    dropoutLayer(null, numberRows, numberColumns, hasFixedLength, random, keepProbability)

fun dropoutLayer(name: String?, numberRows: Int, numberColumns: Int, hasFixedLength: Boolean, random: Random, keepProbability: Float) =
    DropoutLayer(name, numberRows, numberColumns, hasFixedLength, random, keepProbability)