package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuSigmoidLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.activation.CudaSigmoidLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class SigmoidLayer(
    private val name : String?,
    private val numberRows: Int,
    private val numberColumns: Int,
    private val hasFixedLength: Boolean) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuSigmoidLayer(this.name, this.numberRows, if(this.hasFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle) : CudaSigmoidLayer {

        return CudaSigmoidLayer(
            this.name,
            this.numberRows * this.numberColumns,
            { context.createKernel(ForwardKernels.sigmoid()) },
            { context.createKernel(ForwardKernels.backwardSigmoid()) },
            context.maximumNumberOfThreadsPerBlock,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun sigmoidLayer(numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =

    sigmoidLayer(null, numberRows, numberColumns, hasFixedLength)

fun sigmoidLayer(name : String? = null, numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =

    SigmoidLayer(name, numberRows, numberColumns, hasFixedLength)