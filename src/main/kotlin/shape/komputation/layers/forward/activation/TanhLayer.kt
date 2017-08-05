package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuTanhLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.activation.CudaTanhLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class TanhLayer(private val name : String?, private val numberRows : Int, private val numberColumns : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuTanhLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaTanhLayer {

        return CudaTanhLayer(
            this.name,
            this.numberRows * this.numberColumns,
            { context.createKernel(ForwardKernels.tanh()) },
            { context.createKernel(ForwardKernels.backwardTanh()) },
            context.maximumNumberOfThreadsPerBlock,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)


    }

}

fun tanhLayer(numberRows : Int, numberColumns: Int = 1) = tanhLayer(null, numberRows, numberColumns)

fun tanhLayer(name : String? = null, numberRows : Int, numberColumns: Int = 1) = TanhLayer(name, numberRows, numberColumns)