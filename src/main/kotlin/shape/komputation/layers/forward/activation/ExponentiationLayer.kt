package shape.komputation.layers.forward.activation

import jcuda.jcublas.cublasHandle
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.KernelFactory
import shape.komputation.cuda.layers.forward.activation.CudaExponentiationLayer
import shape.komputation.layers.CudaForwardLayerInstruction

class ExponentiationLayer(private val name : String?, private val numberRows : Int, private val numberColumns : Int) : CudaForwardLayerInstruction {

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaExponentiationLayer {

        val kernelFactory = KernelFactory(context.computeCapabilities)

        val forwardExponentiationKernel = kernelFactory.exponentiationKernel()
        val backwardExponentiationKernel = kernelFactory.backwardExponentiationKernel()

        val exponentiationLayer = CudaExponentiationLayer(
            this.name,
            forwardExponentiationKernel,
            backwardExponentiationKernel,
            this.numberRows,
            this.numberColumns)

        return exponentiationLayer

    }

}

fun exponentiationLayer(
    numberRows : Int,
    numberColumns : Int) =

    exponentiationLayer(
        null,
        numberRows,
        numberColumns
    )

fun exponentiationLayer(
    name : String?,
    numberRows : Int,
    numberColumns : Int) =

    ExponentiationLayer(
        name,
        numberRows,
        numberColumns)