package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuIdentityLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaIdentityLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class IdentityLayer(private val name : String?) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuIdentityLayer(this.name)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaIdentityLayer(this.name)
}

fun identityLayer(name : String? = null) =

    IdentityLayer(name)