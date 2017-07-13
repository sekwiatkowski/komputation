package shape.komputation.layers.forward.projection

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.projection.CpuProjectionLayer
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.cpu.optimization.UpdateRule
import shape.komputation.cuda.CudaEnvironment
import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.cuda.layers.forward.projection.CublasProjectionLayer
import shape.komputation.cuda.optimization.CublasUpdateRule
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.optimization.OptimizationInstruction

class ProjectionLayer(
    private val name : String?,
    private val inputDimension: Int,
    private val outputDimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy : OptimizationInstruction? = null) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    override fun buildForCpu(): CpuProjectionLayer {

        val numberWeightRows = this.outputDimension
        val numberWeightColumns = this.inputDimension

        val weights = initializeWeights(this.weightInitializationStrategy, numberWeightRows, numberWeightColumns, this.inputDimension)
        val weightUpdateRule = this.optimizationStrategy?.buildForCpu()?.invoke(numberWeightRows, numberWeightColumns)

        val bias : DoubleArray?
        val biasUpdateRule: UpdateRule?
        val biasAccumulator: DenseAccumulator?

        if (this.biasInitializationStrategy != null) {

            bias = initializeColumnVector(this.biasInitializationStrategy, this.outputDimension)
            biasUpdateRule = this.optimizationStrategy?.buildForCpu()?.invoke(bias.size, 1)
            biasAccumulator = DenseAccumulator(bias.size)

        }
        else {

            bias = null
            biasUpdateRule = null
            biasAccumulator = null

        }

        val weightAccumulator = DenseAccumulator(numberWeightRows * numberWeightColumns)

        return CpuProjectionLayer(this.name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule, bias, biasAccumulator, biasUpdateRule)

    }

    override fun buildForCuda(environment: CudaEnvironment): CudaForwardLayer {

        val cublasHandle = cublasHandle()

        val numberWeightRows = this.outputDimension
        val numberWeightColumns = this.inputDimension

        val weights = initializeWeights(this.weightInitializationStrategy, numberWeightRows, numberWeightColumns, this.inputDimension)
        val weightUpdateRule = this.optimizationStrategy?.buildForCuda()?.invoke(cublasHandle, numberWeightRows, numberWeightColumns)

        val bias : DoubleArray?
        val biasUpdateRule: CublasUpdateRule?

        if (this.biasInitializationStrategy != null) {

            bias = initializeColumnVector(this.biasInitializationStrategy, this.outputDimension)
            biasUpdateRule = this.optimizationStrategy?.buildForCuda()?.invoke(cublasHandle, bias.size, 1)

        }
        else {

            bias = null
            biasUpdateRule = null

        }

        return CublasProjectionLayer(this.name, cublasHandle, weights, numberWeightRows, numberWeightColumns, weightUpdateRule, bias, biasUpdateRule)

    }

}

fun projectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationInstruction? = null) =

    projectionLayer(null, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

fun projectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationInstruction? = null) =

    ProjectionLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)