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
        "dropout/DropoutTrainingKernel.cu")

    fun dropoutRuntime() = createKernel(
        "dropoutRuntimeKernel",
        "dropoutRuntimeKernel",
        "dropout/DropoutRuntimeKernel.cu")

    fun backwardDropout() = createKernel(
        "backwardDropoutKernel",
        "backwardDropoutKernel",
        "dropout/BackwardDropoutKernel.cu")

    fun exponentiation() = createKernel(
        "exponentiationKernel",
        "exponentiationKernel",
        "exponentiation/ExponentiationKernel.cu")

    fun backwardExponentiation() = createKernel(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "exponentiation/BackwardExponentiationKernel.cu")

    fun normalization(blockSize : Int) = createKernel(
        "normalizationKernel",
        "normalizationKernel<$blockSize>",
        "normalization/NormalizationKernel.cu",
        listOf("reduction/Reduction.cuh"))

    fun backwardNormalization(blockSize: Int) = createKernel(
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

    fun squaredLoss(blockSize: Int) = createKernel(
        "squaredLossKernel",
        "squaredLossKernel<$blockSize>",
        "squaredloss/SquaredLossKernel.cu",
        listOf("reduction/Reduction.cuh"))

    fun backwardSquaredLoss() = createKernel(
        "backwardSquaredLossKernel",
        "backwardSquaredLossKernel",
        "squaredloss/BackwardSquaredLossKernel.cu")

    fun logisticLoss(blockSize: Int) = createKernel(
        "logisticLossKernel",
        "logisticLossKernel<$blockSize>",
        "logisticloss/LogisticLossKernel.cu",
        listOf("reduction/Reduction.cuh"))

    fun backwardLogisticLoss() = createKernel(
        "backwardLogisticLossKernel",
        "backwardLogisticLossKernel",
        "logisticloss/BackwardLogisticLossKernel.cu")

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