package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.LossKernels
import shape.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberCategories: Int, private val numberSteps: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss()

    override fun buildForCuda(context: CudaContext): CudaSquaredLoss {

        return CudaSquaredLoss(
            this.numberCategories,
            this.numberSteps,
            { blockSize -> context.createKernel(LossKernels.squaredLoss(blockSize)) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun squaredLoss(numberCategories: Int, numberSteps : Int = 1) =

    SquaredLoss(numberCategories, numberSteps)