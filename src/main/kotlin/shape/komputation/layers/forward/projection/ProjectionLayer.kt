package shape.komputation.layers.forward.projection

import shape.komputation.cpu.forward.projection.CpuProjectionLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.OptimizationStrategy
import shape.komputation.optimization.UpdateRule

class ProjectionLayer(
    private val name : String?,
    private val inputDimension: Int,
    private val outputDimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy : OptimizationStrategy? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuProjectionLayer {

        val numberWeightRows = this.outputDimension
        val numberWeightColumns = this.inputDimension

        val weights = initializeWeights(this.weightInitializationStrategy, numberWeightRows, numberWeightColumns, this.inputDimension)
        val weightUpdateRule = this.optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

        val bias : DoubleArray?
        val biasUpdateRule: UpdateRule?
        val biasAccumulator: DenseAccumulator?

        if (this.biasInitializationStrategy != null) {

            bias = initializeColumnVector(this.biasInitializationStrategy, this.outputDimension)
            biasUpdateRule = this.optimizationStrategy?.invoke(bias.size, 1)
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

}

fun projectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    projectionLayer(null, inputDimension, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

fun projectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    ProjectionLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)