package shape.komputation.layers

import shape.komputation.cuda.CudaEnvironment
import shape.komputation.cuda.layers.CudaForwardLayer

interface CudaForwardLayerInstruction {

    fun buildForCuda(environment : CudaEnvironment) : CudaForwardLayer

}