package shape.komputation.layers.forward.dropout

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.dropout.CpuDropoutLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.dropout.CudaDropoutLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import java.util.*

class DropoutLayer(
    private val name : String?,
    private val numberEntries : Int,
    private val random : Random,
    private val keepProbability : Float) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuDropoutLayer(this.name, this.numberEntries, this.random, this.keepProbability)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaDropoutLayer {

        val kernelFactory = context.kernelFactory

        val trainingKernel = kernelFactory.dropoutTrainingKernel()
        val runtimeKernel = kernelFactory.dropoutRuntimeKernel()
        val backwardKernel = kernelFactory.backwardDropoutKernel()

        val layer = CudaDropoutLayer(this.name, trainingKernel, runtimeKernel, backwardKernel, context.maximumNumberThreadsPerBlock, this.numberEntries, this.random, this.keepProbability)

        return layer

    }

}

fun dropoutLayer(numberEntries: Int, random: Random, keepProbability: Float) =

    dropoutLayer(null, numberEntries, random, keepProbability)

fun dropoutLayer(name: String?, numberEntries: Int, random: Random, keepProbability: Float) =

    DropoutLayer(name, numberEntries, random, keepProbability)