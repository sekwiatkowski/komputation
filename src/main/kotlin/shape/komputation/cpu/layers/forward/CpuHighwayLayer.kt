package shape.komputation.cpu.layers.forward

import shape.komputation.cpu.layers.BaseCpuForwardLayer
import shape.komputation.cpu.layers.combination.AdditionCombination
import shape.komputation.cpu.layers.combination.HadamardCombination
import shape.komputation.cpu.optimization.DenseAccumulator
import shape.komputation.matrix.FloatMatrix
import shape.komputation.matrix.floatColumnVector
import shape.komputation.optimization.Optimizable

class CpuHighwayLayer internal constructor(
    name : String?,
    inputDimension : Int,
    private val transformation : CpuDenseLayer,
    private val transformationFraction : CpuDenseLayer,
    private val transformationHadamard : HadamardCombination,
    private val counterProbability: CpuCounterProbabilityLayer,
    private val carryHadamard : HadamardCombination,
    private val addition : AdditionCombination) : BaseCpuForwardLayer(name), Optimizable {

    private val gradientAccumulator = DenseAccumulator(inputDimension)

    override fun forward(withinBatch : Int, input: FloatMatrix, isTraining : Boolean): FloatMatrix {

        // H(x)
        val transformed = this.transformation.forward(withinBatch, input, isTraining)

        // T(x)
        val transformationFraction = this.transformationFraction.forward(withinBatch, input, isTraining)

        // H(x) (.) T(x)
        val transformationComponent = this.transformationHadamard.forward(transformed, transformationFraction)

        // 1 - T(x)
        val carryFraction = this.counterProbability.forward(withinBatch, transformationFraction, isTraining)

        // x (.) (1 - T(x))
        val carryComponent = this.carryHadamard.forward(input, carryFraction)

        // H(x) (.) T(x) + x (.) (1 - T(x))
        val result = addition.forward(transformationComponent, carryComponent)

        return result

    }

    private fun backwardTransformation(withinBatch : Int, chain: FloatMatrix) {

        // d chain / d H(x) (.) T(x)
        val diffChainWrtTransformationComponent = this.addition.backwardFirst(chain)

        // d H(x) (.) T(x) / d H(x)
        val diffTransformationComponentWrtTransformation = this.transformationHadamard.backwardFirst(diffChainWrtTransformationComponent)

        // d H(x) / d x
        val diffTransformationWrtInput = this.transformation.backward(withinBatch, diffTransformationComponentWrtTransformation)

        this.gradientAccumulator.accumulate(diffTransformationWrtInput.entries)

        // d H(x) (.) T(x) / d T(x)
        val diffTransformationComponentWrtTransformationFraction = this.transformationHadamard.backwardSecond(diffChainWrtTransformationComponent)

        // d T(x) / d x
        val diffTransformationFractionWrtInput = this.transformationFraction.backward(withinBatch, diffTransformationComponentWrtTransformationFraction)

        this.gradientAccumulator.accumulate(diffTransformationFractionWrtInput.entries)

    }

    private fun backwardCarry(withinBatch : Int, chain: FloatMatrix) {

        // d chain / d x (.) (1 - T(x))
        val diffChainWrtCarryComponent = this.addition.backwardSecond(chain)

        // d x (.) (1 - T(x)) / d x
        val diffCarryComponentWrtInput = this.carryHadamard.backwardFirst(diffChainWrtCarryComponent)

        this.gradientAccumulator.accumulate(diffCarryComponentWrtInput.entries)

        // d x (.) (1 - T(x)) / d (1 - T(x))
        val diffCarryComponentWrtCarryFraction = this.carryHadamard.backwardSecond(diffChainWrtCarryComponent)

        // d (1 - T(x)) / d T(x)
        val diffCarryFractionWrtTransformationFraction = this.counterProbability.backward(withinBatch, diffCarryComponentWrtCarryFraction)

        // d T(x) / d x
        val diffTransformationFractionWrtInput = this.transformationFraction.backward(withinBatch, diffCarryFractionWrtTransformationFraction)

        this.gradientAccumulator.accumulate(diffTransformationFractionWrtInput.entries)

    }

    override fun backward(withinBatch : Int, chain: FloatMatrix): FloatMatrix {

        this.backwardTransformation(withinBatch, chain)

        this.backwardCarry(withinBatch, chain)

        val result = floatColumnVector(*this.gradientAccumulator.getAccumulation().copyOf())

        this.gradientAccumulator.reset()

        return result

    }

    override fun optimize(scalingFactor : Float) {

        this.transformation.optimize(scalingFactor)
        this.transformationFraction.optimize(scalingFactor)

    }

}