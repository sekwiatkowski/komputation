package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuExponentiationLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.activation.CudaExponentiationLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction

class ExponentiationLayer(private val name : String?, private val numberRows: Int, private val numberColumns: Int) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuExponentiationLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaExponentiationLayer {

        val exponentiationLayer = CudaExponentiationLayer(
            this.name,
            this.numberRows * this.numberColumns,
            { context.createKernel(ForwardKernels.exponentiation()) },
            { context.createKernel(ForwardKernels.backwardExponentiation()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        return exponentiationLayer

    }

}

fun exponentiationLayer(
    numberRows : Int,
    numberColumns: Int = 1) =

    exponentiationLayer(
        null,
        numberRows,
        numberColumns
    )

fun exponentiationLayer(
    name : String?,
    numberRows : Int,
    numberColumns: Int = 1) =

    ExponentiationLayer(
        name,
        numberRows,
        numberColumns)