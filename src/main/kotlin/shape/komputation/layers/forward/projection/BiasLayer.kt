package shape.komputation.layers.forward.projection

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.projection.CpuBiasLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.layers.forward.projection.CublasBiasLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.optimization.OptimizationInstruction

class BiasLayer(
    private val name : String?,
    private val numberInputRows: Int,
    private val numberInputColumns: Int,
    private val initializationStrategy: InitializationStrategy,
    private val optimizationStrategy : OptimizationInstruction? = null) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu(): CpuBiasLayer {

        val bias = initializeColumnVector(this.initializationStrategy, this.numberInputRows)
        val accumulator = DenseAccumulator(bias.size)
        val updateRule = this.optimizationStrategy?.buildForCpu()?.invoke(this.numberInputRows, this.numberInputColumns)

        val layer = CpuBiasLayer(this.name, bias, accumulator, updateRule)

        return layer

    }

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CublasBiasLayer {

        val bias = initializeColumnVector(this.initializationStrategy, this.numberInputRows)
        val updateRule = this.optimizationStrategy?.buildForCuda(context)?.invoke(1, this.numberInputRows, this.numberInputColumns)

        val layer = CublasBiasLayer(
            this.name,
            cublasHandle,
            context.maximumNumberOfThreadsPerBlock,
            this.numberInputRows,
            this.numberInputColumns,
            { context.createKernel(ForwardKernels.bias()) }, bias, updateRule)

        return layer

    }

}

fun biasLayer(
    numberRows : Int,
    numberColumns : Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =

    biasLayer(null, numberRows, numberColumns, initializationStrategy, optimizationStrategy)

fun biasLayer(
    name : String?,
    numberRows : Int,
    numberColumns : Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =

    BiasLayer(name, numberRows, numberColumns, initializationStrategy, optimizationStrategy)