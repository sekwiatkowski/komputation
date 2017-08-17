package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuReluLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.activation.CudaReluLayer
import shape.komputation.layers.CpuActivationLayerInstruction
import shape.komputation.layers.CudaActivationLayerInstruction

class ReluLayer(
    private val name : String?,
    private val numberRows : Int,
    private val numberColumns : Int,
    private val hasFixedLength:
    Boolean) : CpuActivationLayerInstruction, CudaActivationLayerInstruction {

    override fun buildForCpu() =

        CpuReluLayer(this.name, this.numberRows, if(this.hasFixedLength) this.numberColumns else 1, this.numberColumns)

    override fun buildForCuda(context : CudaContext, cublasHandle: cublasHandle): CudaReluLayer {

        return CudaReluLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { context.createKernel(ForwardKernels.relu()) },
            { context.createKernel(ForwardKernels.backwardRelu()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

    }

}

fun reluLayer(numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =

    reluLayer(null, numberRows, numberColumns, hasFixedLength)

fun reluLayer(name : String? = null, numberRows : Int, numberColumns: Int = 1, hasFixedLength: Boolean = true) =

    ReluLayer(name, numberRows, numberColumns, hasFixedLength)