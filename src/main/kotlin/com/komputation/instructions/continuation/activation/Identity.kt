package com.komputation.instructions.continuation.activation

import com.komputation.cpu.layers.continuation.activation.CpuIdentity
import com.komputation.cuda.CudaContext
import com.komputation.cuda.layers.continuation.activation.CudaIdentity
import com.komputation.instructions.continuation.ActivationInstruction
import com.komputation.instructions.continuation.BaseEntrywiseInstruction
import jcuda.jcublas.cublasHandle

class Identity internal constructor(private val name : String?) : BaseEntrywiseInstruction(), ActivationInstruction {

    override fun buildForCpu() =
        CpuIdentity(this.name, this.numberInputRows, this.minimumNumberInputColumns, this.maximumNumberInputColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaIdentity(this.name, this.numberInputRows, this.maximumNumberInputColumns)
}

fun identityLayer(name : String? = null) =
    Identity(name)