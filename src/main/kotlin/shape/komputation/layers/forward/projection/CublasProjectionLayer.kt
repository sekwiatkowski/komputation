package shape.komputation.layers.forward.projection

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.forward.projection.TempCublasProjectionLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.optimization.CublasOptimizationStrategy
import shape.komputation.optimization.CublasUpdateRule

class CublasProjectionLayer(
    private val name : String?,
    private val inputDimension: Int,
    private val outputDimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy : CublasOptimizationStrategy?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): TempCublasProjectionLayer {

        val cublasHandle = cublasHandle()

        val numberWeightRows = this.outputDimension
        val numberWeightColumns = this.inputDimension

        val weights = initializeWeights(this.weightInitializationStrategy, numberWeightRows, numberWeightColumns, this.inputDimension)
        val weightUpdateRule = this.optimizationStrategy?.invoke(cublasHandle, numberWeightRows, numberWeightColumns)

        val bias : DoubleArray?
        val biasUpdateRule: CublasUpdateRule?

        if (this.biasInitializationStrategy != null) {

            bias = initializeColumnVector(this.biasInitializationStrategy, this.outputDimension)
            biasUpdateRule = this.optimizationStrategy?.invoke(cublasHandle, bias.size, 1)

        }
        else {

            bias = null
            biasUpdateRule = null

        }

        return TempCublasProjectionLayer(this.name, cublasHandle, weights, numberWeightRows, numberWeightColumns, weightUpdateRule, bias, biasUpdateRule)

    }

}

fun cublasProjectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : CublasOptimizationStrategy? = null) =

    cublasProjectionLayer(
        null,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy
    )


fun cublasProjectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : CublasOptimizationStrategy? = null) =

    CublasProjectionLayer(name, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)