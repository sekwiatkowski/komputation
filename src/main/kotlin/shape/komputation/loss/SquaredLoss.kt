package shape.komputation.loss

import shape.komputation.cpu.loss.CpuSquaredLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.LossKernels
import shape.komputation.cuda.loss.CudaSquaredLoss

class SquaredLoss(private val numberRows: Int, private val numberColumns: Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuSquaredLoss(this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext): CudaSquaredLoss {

        return CudaSquaredLoss(
            this.numberRows,
            this.numberColumns,
            { blockSize -> context.createKernel(LossKernels.squaredLoss(blockSize)) },
            { context.createKernel(LossKernels.backwardSquaredLoss()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun squaredLoss(numberCategories: Int, length: Int = 1) =

    SquaredLoss(numberCategories, length)