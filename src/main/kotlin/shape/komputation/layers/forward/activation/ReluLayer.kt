package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuReluLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaReluLayer
import shape.komputation.layers.CpuDropoutCompliantInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class ReluLayer(private val name : String?, private val numberEntries : Int) : CpuDropoutCompliantInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuReluLayer(this.name, this.numberEntries)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle): CudaReluLayer {

        val kernelFactory = context.kernelFactory

        val forwardKernel = kernelFactory.relu()
        val backwardKernel = kernelFactory.backwardRelu()

        return CudaReluLayer(this.name, forwardKernel, backwardKernel, context.maximumNumberThreadsPerBlock, this.numberEntries)

    }

}

fun reluLayer(numberEntries : Int) = reluLayer(null, numberEntries)

fun reluLayer(name : String? = null, numberEntries : Int) = ReluLayer(name, numberEntries)