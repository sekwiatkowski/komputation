package shape.komputation.layers.forward.convolution

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.maxpooling.CpuMaxPoolingLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.maxpooling.CudaMaxPoolingLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction

class MaxPoolingLayer internal constructor (
    private val name : String?,
    private val numberRows : Int,
    private val minimumColumns : Int,
    private val maximumColumns : Int,
    private val symbolForUnusedColumns : Float) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuMaxPoolingLayer(this.name, this.numberRows, this.minimumColumns, this.maximumColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =

        CudaMaxPoolingLayer(
            this.name,
            this.numberRows,
            this.maximumColumns,
            this.symbolForUnusedColumns,
            { context.createKernel(ForwardKernels.maxPooling()) },
            { context.createKernel(ForwardKernels.backwardMaxPooling()) },
            context.maximumNumberOfThreadsPerBlock,
            context.warpSize)

}

fun maxPoolingLayer(numberRows : Int, numberColumns: Int) =

    maxPoolingLayer(null, numberRows, numberColumns)

fun maxPoolingLayer(name : String? = null, numberRows : Int, numberColumns: Int) =

    MaxPoolingLayer(name, numberRows, numberColumns, numberColumns, Float.NaN)