package shape.komputation.loss

import shape.komputation.cpu.loss.CpuLogisticLoss
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.loss.CudaLogisticLoss

class LogisticLoss(private val numberCategories : Int, private val numberSteps : Int) : CpuLossFunctionInstruction, CudaLossFunctionInstruction {

    override fun buildForCpu() =

        CpuLogisticLoss()

    override fun buildForCuda(context: CudaContext): CudaLogisticLoss {

        val kernelFactory = context.kernelFactory

        val blockSize = Math.pow(2.0, Math.ceil(Math.log(this.numberCategories.toDouble()) / Math.log(2.0))).toInt()

        val forwardKernel = kernelFactory.logisticLoss(blockSize)
        val backwardKernel = kernelFactory.backwardLogisticLoss()

        return CudaLogisticLoss(forwardKernel, backwardKernel, this.numberCategories, this.numberSteps, blockSize)

    }

}

fun logisticLoss(numberCategories: Int, numberSteps: Int = 1) =

    LogisticLoss(numberCategories, numberSteps)