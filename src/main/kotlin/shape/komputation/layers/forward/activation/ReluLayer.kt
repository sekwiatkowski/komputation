package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuReluLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.KernelFactory
import shape.komputation.cuda.layers.forward.activation.CudaReluLayer
import shape.komputation.layers.CpuDropoutCompliantInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class ReluLayer(private val name : String?, private val numberEntries : Int) : CpuDropoutCompliantInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuReluLayer(this.name)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle): CudaReluLayer {

        val kernelFactory = KernelFactory(context.computeCapabilities)

        val reluKernel = kernelFactory.relu()
        val backwardReluKernel = kernelFactory.backwardRelu()

        return CudaReluLayer(this.name, reluKernel, backwardReluKernel, context.maximumNumberThreadsPerBlock, numberEntries)

    }

}

fun reluLayer(numberEntries : Int) = reluLayer(null, numberEntries)

fun reluLayer(name : String? = null, numberEntries : Int) = ReluLayer(name, numberEntries)