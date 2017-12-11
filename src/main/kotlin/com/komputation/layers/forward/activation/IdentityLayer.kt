package com.komputation.layers.forward.activation

import com.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.forward.activation.CudaIdentityLayer
import jcuda.jcublas.cublasHandle

class IdentityLayer internal constructor(
    private val name : String?,
    private val numberRows : Int,
    private val minimumColumns: Int,
    private val maximumColumns : Int) : ActivationLayerInstruction {

    override fun buildForCpu() =
        CpuIdentityLayer(this.name, this.numberRows, this.minimumColumns, this.maximumColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaIdentityLayer(this.name, this.numberRows, this.maximumColumns)
}

fun identityLayer(name : String? = null, numberRows : Int, minimumColumns: Int = 1, maximumColumns: Int = 1) =
    IdentityLayer(name, numberRows, minimumColumns, maximumColumns)