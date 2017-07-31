package shape.komputation.layers.forward

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.CpuNormalizationLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.CudaNormalizationLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction

class NormalizationLayer(private val name : String?, private val numberRows : Int, private val numberColumns : Int) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuNormalizationLayer(this.name, this.numberRows, this.numberColumns)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaNormalizationLayer {

        val kernelFactory = context.kernelFactory

        val normalizationLayer = CudaNormalizationLayer(
            this.name,
            this.numberRows,
            this.numberColumns,
            { blockSize -> kernelFactory.normalization(blockSize) },
            { blockSize -> kernelFactory.backwardNormalization(blockSize) },
            context.maximumNumberOfThreadsPerBlock)

        return normalizationLayer

    }

}

fun normalizationLayer(
    numberRows : Int,
    numberColumns : Int) =

    normalizationLayer(
        null,
        numberRows,
        numberColumns
    )

fun normalizationLayer(
    name : String?,
    numberRows : Int,
    numberColumns : Int) =

    NormalizationLayer(
        name,
        numberRows,
        numberColumns)