package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val dimension : Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

    override fun buildForCuda(context: CudaContext) =

        CudaSquaredLoss(context.computeCapabilities, dimension)

}

fun squaredLoss(dimension: Int) =

    SquaredLoss(dimension)