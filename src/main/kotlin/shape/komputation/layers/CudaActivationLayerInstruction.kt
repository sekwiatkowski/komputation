package shape.komputation.layers

import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaActivationLayer

interface CudaActivationLayerInstruction : CudaForwardLayerInstruction {

    override fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle): CudaActivationLayer

}