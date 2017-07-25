package shape.komputation.layers.forward.dropout

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.dropout.CpuDropoutLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.cuda.layers.forward.dropout.CudaDropoutLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import java.util.*

class DropoutLayer(
    private val name : String?,
    private val random: Random,
    private val numberEntries: Int,
    private val keepProbability: Float) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuDropoutLayer(this.name, this.random, this.numberEntries, this.keepProbability)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaDropoutLayer {

        val dropoutTrainingKernel = context.kernelFactory.dropoutTrainingKernel()

        val layer = CudaDropoutLayer(this.name, dropoutTrainingKernel)

        return layer

    }

}

fun dropoutLayer(random: Random, numberEntries: Int, keepProbability: Float) =

    dropoutLayer(null, random, numberEntries, keepProbability)

fun dropoutLayer(name: String?, random: Random, numberEntries: Int, keepProbability: Float) =

    DropoutLayer(name, random, numberEntries, keepProbability)