package shape.komputation.loss

import shape.komputation.cuda.CudaEnvironment
import shape.komputation.cuda.loss.CudaLossFunction

interface CudaLossFunctionInstruction {

    fun buildForCuda(environment: CudaEnvironment) : CudaLossFunction

}