package shape.komputation.layers.forward

import shape.komputation.cpu.combination.additionCombination
import shape.komputation.cpu.combination.hadamardCombination
import shape.komputation.cpu.forward.CpuHighwayLayer
import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.optimization.OptimizationStrategy

class HighwayLayer(
    private val name : String?,
    private val dimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val transformationBiasInitializationStrategy: InitializationStrategy?,
    private val transformationFractionBiasInitializationStrategy: InitializationStrategy?,
    private val transformationFunction: ActivationFunction,
    private val optimizationStrategy: OptimizationStrategy?) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuHighwayLayer {

        val transformation = denseLayer(this.name, this.dimension, this.dimension, this.weightInitializationStrategy, this.transformationBiasInitializationStrategy, this.transformationFunction, this.optimizationStrategy).buildForCpu()

        val transformationFraction = denseLayer(this.name, this.dimension, this.dimension, this.weightInitializationStrategy, this.transformationFractionBiasInitializationStrategy, ActivationFunction.Sigmoid, this.optimizationStrategy).buildForCpu()

        val transformationHadamard = hadamardCombination(this.name)

        val counterProbability = counterProbabilityLayer(this.name, this.dimension).buildForCpu()

        val carryHadamard = hadamardCombination(this.name)

        val addition = additionCombination(this.name)

        val highwayLayer = CpuHighwayLayer(this.name, this.dimension, transformation, transformationFraction, transformationHadamard, counterProbability, carryHadamard, addition)

        return highwayLayer

    }

}

fun highwayLayer(
    dimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    transformationBiasInitializationStrategy: InitializationStrategy?,
    transformationFractionBiasInitializationStrategy: InitializationStrategy?,
    transformationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    highwayLayer(null, dimension, weightInitializationStrategy, transformationBiasInitializationStrategy, transformationFractionBiasInitializationStrategy, transformationFunction, optimizationStrategy)

fun highwayLayer(
    name : String?,
    dimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    transformationBiasInitializationStrategy: InitializationStrategy?,
    transformationFractionBiasInitializationStrategy: InitializationStrategy?,
    transformationFunction: ActivationFunction,
    optimizationStrategy: OptimizationStrategy?) =

    HighwayLayer(
        name,
        dimension,
        weightInitializationStrategy,
        transformationBiasInitializationStrategy,
        transformationFractionBiasInitializationStrategy,
        transformationFunction,
        optimizationStrategy
    )