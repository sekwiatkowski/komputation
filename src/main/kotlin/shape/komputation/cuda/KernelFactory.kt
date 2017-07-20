package shape.komputation.cuda

import java.io.File

class KernelFactory(private val capabilities : Pair<Int, Int>) {

    fun exponentiationKernel() = createKernel(
        "exponentiationKernel",
        "exponentiationKernel",
        "exponentiation/ExponentiationKernel.cu")

    fun backwardExponentiationKernel() = createKernel(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "exponentiation/BackwardExponentiationKernel.cu")

    fun projectionKernel() = createKernel(
        "projectionKernel",
        "projectionKernel",
        "projection/ProjectionKernel.cu",
        listOf("projection/Projection.cuh"))

    fun projectionWithBiasKernel() = createKernel(
        "projectionWithBiasKernel",
        "projectionWithBiasKernel",
        "projection/ProjectionWithBiasKernel.cu",
        listOf("projection/Projection.cuh"))

    fun accumulationKernel() = createKernel(
        "accumulationKernel",
        "accumulationKernel",
        "accumulation/AccumulationKernel.cu")

    fun normalizationKernel(blockSize : Int) =

        createKernel(
            "normalizationKernel",
            "normalizationKernel<$blockSize>",
            "normalization/NormalizationKernel.cu",
            listOf("reduction/Reduction.cuh"))

    fun backwardNormalizationKernel(blockSize: Int) =

        createKernel(
            "backwardNormalizationKernel",
            "backwardNormalizationKernel<$blockSize>",
            "normalization/BackwardNormalizationKernel.cu",
            listOf("reduction/Reduction.cuh"))

    fun sigmoid() = createKernel(
        "sigmoidKernel",
        "sigmoidKernel",
        "sigmoid/SigmoidKernel.cu")

    fun backwardSigmoid() = createKernel(
        "backwardSigmoidKernel",
        "backwardSigmoidKernel",
        "sigmoid/BackwardSigmoidKernel.cu")

    fun relu() = createKernel(
        "reluKernel",
        "reluKernel",
        "relu/ReluKernel.cu")

    fun backwardRelu() = createKernel(
        "backwardReluKernel",
        "backwardReluKernel",
        "relu/BackwardReluKernel.cu")

    fun tanh() = createKernel(
        "tanhKernel",
        "tanhKernel",
        "tanh/TanhKernel.cu")

    fun backwardTanh() = createKernel(
        "backwardTanhKernel",
        "backwardTanhKernel",
        "tanh/BackwardTanhKernel.cu")

    fun stochasticGradientDescent() = createKernel(
        "stochasticGradientDescentKernel",
        "stochasticGradientDescentKernel",
        "stochasticgradientdescent/StochasticGradientDescentKernel.cu")

    fun squaredLoss() = createKernel(
        "squaredLossKernel",
        "squaredLossKernel",
        "squaredloss/SquaredLossKernel.cu")

    fun backwardSquaredLoss() = createKernel(
        "backwardSquaredLossKernel",
        "backwardSquaredLossKernel",
        "squaredloss/BackwardSquaredLossKernel.cu")

    private fun resolveRelativePath(relativePath: String)  =

        File(this.javaClass.getResource("/cuda/$relativePath").toURI())

    private fun createKernel(name : String, nameExpression : String, relativePath : String, relativeHeaderPaths: List<String> = emptyList()): Kernel {

        val kernelFile = resolveRelativePath(relativePath)

        val includeNames = Array(relativeHeaderPaths.size) { index ->

            relativeHeaderPaths[index]

        }

        val headerFiles = Array(relativeHeaderPaths.size) { index -> resolveRelativePath(relativeHeaderPaths[index]) }

        return Kernel(this.capabilities, kernelFile, name, nameExpression, headerFiles, includeNames)

    }

}