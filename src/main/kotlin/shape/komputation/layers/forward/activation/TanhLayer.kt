package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuTanhLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaTanhLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class TanhLayer(private val name : String?, private val numberEntries : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuTanhLayer(this.name)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaTanhLayer {

        val kernelFactory = context.kernelFactory

        val forwardKernel = kernelFactory.tanh()
        val backwardKernel = kernelFactory.backwardTanh()

        return CudaTanhLayer(this.name, forwardKernel, backwardKernel, context.maximumNumberThreadsPerBlock, this.numberEntries)


    }

}

fun tanhLayer(numberEntries : Int) = tanhLayer(null, numberEntries)

fun tanhLayer(name : String? = null, numberEntries : Int) = TanhLayer(name, numberEntries)