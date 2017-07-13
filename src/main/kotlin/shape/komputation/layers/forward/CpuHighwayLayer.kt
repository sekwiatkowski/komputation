package shape.komputation.layers.forward

import shape.komputation.functions.activation.ActivationFunction
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.BaseForwardLayer
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.combination.AdditionCombination
import shape.komputation.layers.combination.HadamardCombination
import shape.komputation.layers.combination.additionCombination
import shape.komputation.layers.combination.hadamardCombination
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.matrix.doubleColumnVector
import shape.komputation.optimization.DenseAccumulator
import shape.komputation.optimization.Optimizable
import shape.komputation.optimization.OptimizationStrategy

class CpuHighwayLayer internal constructor(
    name : String?,
    inputDimension : Int,
    private val transformation : CpuDenseLayer,
    private val transformationFraction : CpuDenseLayer,
    private val transformationHadamard : HadamardCombination,
    private val counterProbability: CpuCounterProbabilityLayer,
    private val carryHadamard : HadamardCombination,
    private val addition : AdditionCombination) : BaseForwardLayer(name), Optimizable {

    private val gradientAccumulator = DenseAccumulator(inputDimension)

    override fun forward(input: DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        // H(x)
        val transformed = this.transformation.forward(input, isTraining)

        // T(x)
        val transformationFraction = this.transformationFraction.forward(input, isTraining)

        // H(x) (.) T(x)
        val transformationComponent = this.transformationHadamard.forward(transformed, transformationFraction)

        // 1 - T(x)
        val carryFraction = this.counterProbability.forward(transformationFraction, isTraining)

        // x (.) (1 - T(x))
        val carryComponent = this.carryHadamard.forward(input, carryFraction)

        // H(x) (.) T(x) + x (.) (1 - T(x))
        val result = addition.forward(transformationComponent, carryComponent)

        return result

    }

    private fun backwardTransformation(chain: DoubleMatrix) {

        // d chain / d H(x) (.) T(x)
        val diffChainWrtTransformationComponent = this.addition.backwardFirst(chain)

        // d H(x) (.) T(x) / d H(x)
        val diffTransformationComponentWrtTransformation = this.transformationHadamard.backwardFirst(diffChainWrtTransformationComponent)

        // d H(x) / d x
        val diffTransformationWrtInput = this.transformation.backward(diffTransformationComponentWrtTransformation)

        this.gradientAccumulator.accumulate(diffTransformationWrtInput.entries)

        // d H(x) (.) T(x) / d T(x)
        val diffTransformationComponentWrtTransformationFraction = this.transformationHadamard.backwardSecond(diffChainWrtTransformationComponent)

        // d T(x) / d x
        val diffTransformationFractionWrtInput = this.transformationFraction.backward(diffTransformationComponentWrtTransformationFraction)

        this.gradientAccumulator.accumulate(diffTransformationFractionWrtInput.entries)

    }

    private fun backwardCarry(chain: DoubleMatrix) {

        // d chain / d x (.) (1 - T(x))
        val diffChainWrtCarryComponent = this.addition.backwardSecond(chain)

        // d x (.) (1 - T(x)) / d x
        val diffCarryComponentWrtInput = this.carryHadamard.backwardFirst(diffChainWrtCarryComponent)

        this.gradientAccumulator.accumulate(diffCarryComponentWrtInput.entries)

        // d x (.) (1 - T(x)) / d (1 - T(x))
        val diffCarryComponentWrtCarryFraction = this.carryHadamard.backwardSecond(diffChainWrtCarryComponent)

        // d (1 - T(x)) / d T(x)
        val diffCarryFractionWrtTransformationFraction = this.counterProbability.backward(diffCarryComponentWrtCarryFraction)

        // d T(x) / d x
        val diffTransformationFractionWrtInput = this.transformationFraction.backward(diffCarryFractionWrtTransformationFraction)

        this.gradientAccumulator.accumulate(diffTransformationFractionWrtInput.entries)

    }

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        this.backwardTransformation(chain)

        this.backwardCarry(chain)

        val result = doubleColumnVector(*this.gradientAccumulator.getAccumulation().copyOf())

        this.gradientAccumulator.reset()

        return result

    }

    override fun optimize(scalingFactor : Double) {

        this.transformation.optimize(scalingFactor)
        this.transformationFraction.optimize(scalingFactor)

    }

}

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

        val highwayLayer =CpuHighwayLayer(this.name, this.dimension, transformation, transformationFraction, transformationHadamard, counterProbability, carryHadamard, addition)

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