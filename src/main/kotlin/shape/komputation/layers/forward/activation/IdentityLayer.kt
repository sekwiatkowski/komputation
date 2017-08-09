package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaIdentityLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class IdentityLayer(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaIdentityLayer(this.name)
}

fun identityLayer(name : String? = null, numberRows : Int, numberColumns : Int = 1) =

    IdentityLayer(name, numberRows, numberColumns)