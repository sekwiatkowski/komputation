package com.komputation.layers.forward.activation

import com.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.forward.activation.CudaIdentityLayer
import jcuda.jcublas.cublasHandle

class IdentityLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int) : ActivationLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaIdentityLayer(this.name, this.numberRows, this.numberColumns)
}

fun identityLayer(name : String? = null, numberRows : Int, numberColumns : Int = 1) =

    IdentityLayer(name, numberRows, numberColumns)