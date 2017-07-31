package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuTanhLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.activation.CudaTanhLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class TanhLayer(private val name : String?, private val numberEntries : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuTanhLayer(this.name, this.numberEntries)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaTanhLayer {

        return CudaTanhLayer(
            this.name,
            this.numberEntries,
            { context.createKernel(ForwardKernels.tanh()) },
            { context.createKernel(ForwardKernels.backwardTanh()) },
            context.maximumNumberOfThreadsPerBlock,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)


    }

}

fun tanhLayer(numberEntries : Int) = tanhLayer(null, numberEntries)

fun tanhLayer(name : String? = null, numberEntries : Int) = TanhLayer(name, numberEntries)