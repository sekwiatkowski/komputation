package shape.komputation.optimization

import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.optimization.CudaOptimizationStrategy

interface CudaOptimizationInstruction {

    fun buildForCuda(context: CudaContext) : CudaOptimizationStrategy

}