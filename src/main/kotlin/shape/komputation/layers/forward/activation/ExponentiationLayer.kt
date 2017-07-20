package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.KernelFactory
import shape.komputation.cuda.layers.forward.activation.CudaExponentiationLayer
import shape.komputation.layers.CudaForwardLayerInstruction

class ExponentiationLayer(private val name : String?, private val numberEntries: Int) : CudaForwardLayerInstruction {

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaExponentiationLayer {

        val kernelFactory = KernelFactory(context.computeCapabilities)

        val forwardExponentiationKernel = kernelFactory.exponentiationKernel()
        val backwardExponentiationKernel = kernelFactory.backwardExponentiationKernel()

        val exponentiationLayer = CudaExponentiationLayer(
            this.name,
            forwardExponentiationKernel,
            backwardExponentiationKernel,
            context.maximumNumberThreadsPerBlock,
            this.numberEntries)

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