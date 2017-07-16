package shape.komputation.loss

import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.loss.CudaLossFunction

interface CudaLossFunctionInstruction {

    fun buildForCuda(context: CudaContext) : CudaLossFunction

}