package shape.komputation.cuda

import java.io.File

class KernelFactory(private val capabilities : Pair<Int, Int>) {

    fun bias() = createKernel(
        "biasKernel",
        "biasKernel",
        "bias/BiasKernel.cu")

    fun dropoutTraining() = createKernel(
        "dropoutTrainingKernel",
        "dropoutTrainingKernel",
        "dropout/DropoutTrainingKernel.cu",
        listOf("zero/Zero.cuh"))

    fun dropoutRuntime() = createKernel(
        "dropoutRuntimeKernel",
        "dropoutRuntimeKernel",
        "dropout/DropoutRuntimeKernel.cu",
        listOf("zero/Zero.cuh"))

    fun backwardDropout() = createKernel(
        "backwardDropoutKernel",
        "backwardDropoutKernel",
        "dropout/BackwardDropoutKernel.cu",
        listOf("zero/Zero.cuh"))

    fun exponentiation() = createKernel(
        "exponentiationKernel",
        "exponentiationKernel",
        "exponentiation/ExponentiationKernel.cu",
        listOf("zero/Zero.cuh"))

    fun backwardExponentiation() = createKernel(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "exponentiation/BackwardExponentiationKernel.cu",
        listOf("zero/Zero.cuh"))

    fun normalization(blockSize : Int) = createKernel(
        "normalizationKernel",
        "normalizationKernel<$blockSize>",
        "normalization/NormalizationKernel.cu",
        listOf("reduction/Reduction.cuh", "zero/Zero.cuh"))

    fun backwardNormalization(blockSize: Int) = createKernel(
        "backwardNormalizationKernel",
        "backwardNormalizationKernel<$blockSize>",
        "normalization/BackwardNormalizationKernel.cu",
        listOf("reduction/Reduction.cuh", "zero/Zero.cuh"))

    fun sigmoid() = createKernel(
        "sigmoidKernel",
        "sigmoidKernel",
        "sigmoid/SigmoidKernel.cu",
        listOf("zero/Zero.cuh"))

    fun backwardSigmoid() = createKernel(
        "backwardSigmoidKernel",
        "backwardSigmoidKernel",
        "sigmoid/BackwardSigmoidKernel.cu",
        listOf("zero/Zero.cuh"))

    fun relu() = createKernel(
        "reluKernel",
        "reluKernel",
        "relu/ReluKernel.cu",
        listOf("zero/Zero.cuh"))

    fun backwardRelu() = createKernel(
        "backwardReluKernel",
        "backwardReluKernel",
        "relu/BackwardReluKernel.cu",
        listOf("zero/Zero.cuh"))

    fun tanh() = createKernel(
        "tanhKernel",
        "tanhKernel",
        "tanh/TanhKernel.cu",
        listOf("zero/Zero.cuh"))

    fun backwardTanh() = createKernel(
        "backwardTanhKernel",
        "backwardTanhKernel",
        "tanh/BackwardTanhKernel.cu",
        listOf("zero/Zero.cuh"))

    fun stochasticGradientDescent() = createKernel(
        "stochasticGradientDescentKernel",
        "stochasticGradientDescentKernel",
        "stochasticgradientdescent/StochasticGradientDescentKernel.cu")

    fun squaredLoss(blockSize: Int) = createKernel(
        "squaredLossKernel",
        "squaredLossKernel<$blockSize>",
        "squaredloss/SquaredLossKernel.cu",
        listOf("reduction/Reduction.cuh"))

    fun backwardSquaredLoss() = createKernel(
        "backwardSquaredLossKernel",
        "backwardSquaredLossKernel",
        "squaredloss/BackwardSquaredLossKernel.cu",
        listOf("zero/Zero.cuh"))

    fun logisticLoss(blockSize: Int) = createKernel(
        "logisticLossKernel",
        "logisticLossKernel<$blockSize>",
        "logisticloss/LogisticLossKernel.cu",
        listOf("reduction/Reduction.cuh"))

    fun backwardLogisticLoss() = createKernel(
        "backwardLogisticLossKernel",
        "backwardLogisticLossKernel",
        "logisticloss/BackwardLogisticLossKernel.cu",
        listOf("zero/Zero.cuh"))

    private fun resolveRelativePath(relativePath: String) =

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