package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuSigmoidLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaSigmoidLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class SigmoidLayer(private val name : String?, private val numberEntries: Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuSigmoidLayer(this.name)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) : CudaSigmoidLayer {

        val kernelFactory = context.kernelFactory

        val forwardKernel = kernelFactory.sigmoid()
        val backwardKernel = kernelFactory.backwardSigmoid()

        return CudaSigmoidLayer(name, forwardKernel, backwardKernel, context.maximumNumberThreadsPerBlock, this.numberEntries)

    }

}

fun sigmoidLayer(inputDimension: Int) =

    SigmoidLayer(null, inputDimension)

fun sigmoidLayer(name : String? = null, inputDimension: Int) =

    SigmoidLayer(name, inputDimension)