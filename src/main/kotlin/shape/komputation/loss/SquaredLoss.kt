package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val dimension : Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

    override fun buildForCuda(context: CudaContext): CudaSquaredLoss {

        val kernelFactory = context.kernelFactory

        val blockSize = Math.pow(2.0, Math.ceil(Math.log(this.dimension.toDouble()) / Math.log(2.0))).toInt()

        return CudaSquaredLoss(kernelFactory.squaredLoss(blockSize), kernelFactory.backwardSquaredLoss(), this.dimension, blockSize)

    }

}

fun squaredLoss(dimension: Int) =

    SquaredLoss(dimension)