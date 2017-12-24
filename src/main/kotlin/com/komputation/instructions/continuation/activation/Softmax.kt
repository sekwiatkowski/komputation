package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuSoftmax
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.continuation.activation.CudaSoftmax
import com.komputation.instructions.*
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.instructions.continuation.normalization.normalization
import jcuda.jcublas.cublasHandle

class Softmax internal constructor(private val name : String?) : BaseHigherOrderInstruction(), ActivationInstruction {

    private val exponentiationLayer = exponentiation(concatenateNames(this.name, "exponentiation"))
    private val normalizationLayer = normalization(concatenateNames(this.name, "normalization"))

    override fun getLayers() = arrayOf(this.exponentiationLayer, this.normalizationLayer)

    override fun buildForCpu() =
        CpuSoftmax(this.name, this.exponentiationLayer.buildForCpu(), this.normalizationLayer.buildForCpu())

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaSoftmax(this.name, this.exponentiationLayer.buildForCuda(context, cublasHandle), this.normalizationLayer.buildForCuda(context, cublasHandle))

}

fun softmax(name : String? = null) =
    Softmax(name)