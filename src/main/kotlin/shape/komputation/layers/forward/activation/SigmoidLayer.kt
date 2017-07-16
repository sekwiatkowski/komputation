package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuSigmoidLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaSigmoidLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class SigmoidLayer(private val name : String?, private val inputDimension : Int) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuSigmoidLayer(this.name)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) =

        CudaSigmoidLayer(name, context.computeCapabilities, context.numberThreadsPerBlock, this.inputDimension)

}

fun sigmoidLayer(inputDimension: Int) =

    SigmoidLayer(null, inputDimension)

fun sigmoidLayer(name : String? = null, inputDimension: Int) =

    SigmoidLayer(name, inputDimension)