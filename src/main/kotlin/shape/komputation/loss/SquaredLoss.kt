package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberCategories: Int, private val numberSteps: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

    override fun buildForCuda(context: CudaContext): CudaSquaredLoss {

        val kernelFactory = context.kernelFactory

        return CudaSquaredLoss(
            this.numberCategories,
            this.numberSteps,
            { blockSize -> kernelFactory.squaredLoss(blockSize) },
            { kernelFactory.backwardSquaredLoss() },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun squaredLoss(numberCategories: Int, numberSteps : Int = 1) =

    SquaredLoss(numberCategories, numberSteps)