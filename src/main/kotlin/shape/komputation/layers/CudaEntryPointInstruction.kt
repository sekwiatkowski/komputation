package shape.komputation.layers

import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.CudaEntryPoint

interface CudaEntryPointInstruction {

    fun buildForCuda(context : CudaContext) : CudaEntryPoint

}