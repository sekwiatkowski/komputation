package shape.komputation.optimization

import shape.komputation.cuda.optimization.CudaOptimizationStrategy

interface CudaOptimizationInstruction {

    fun buildForCuda() : CudaOptimizationStrategy

}