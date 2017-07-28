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

        val blockSize = Math.pow(2.0, Math.ceil(Math.log(this.numberRows.toDouble()) / Math.log(2.0))).toInt()

        val normalizationLayer = CudaNormalizationLayer(
            this.name,
            { kernelFactory.normalization(blockSize) },
            { kernelFactory.backwardNormalization(blockSize) },
            blockSize,
            this.numberRows,
            this.numberColumns)

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