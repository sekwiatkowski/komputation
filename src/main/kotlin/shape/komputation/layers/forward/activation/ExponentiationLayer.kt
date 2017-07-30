package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.activation.CpuExponentiationLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.forward.activation.CudaExponentiationLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction

class ExponentiationLayer(private val name : String?, private val numberEntries: Int) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu() =

        CpuExponentiationLayer(this.name, this.numberEntries)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaExponentiationLayer {

        val kernelFactory = context.kernelFactory

        val exponentiationLayer = CudaExponentiationLayer(
            this.name,
            this.numberEntries,
            { kernelFactory.exponentiation() },
            { kernelFactory.backwardExponentiation() },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        return exponentiationLayer

    }

}

fun exponentiationLayer(
    numberEntries : Int) =

    exponentiationLayer(
        null,
        numberEntries
    )

fun exponentiationLayer(
    name : String?,
    numberEntries : Int) =

    ExponentiationLayer(
        name,
        numberEntries)