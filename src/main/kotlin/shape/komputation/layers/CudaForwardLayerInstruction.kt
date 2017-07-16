package shape.komputation.layers

import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.CudaForwardLayer

interface CudaForwardLayerInstruction {

    fun buildForCuda(context : CudaContext, cublasHandle : cublasHandle) : CudaForwardLayer

}