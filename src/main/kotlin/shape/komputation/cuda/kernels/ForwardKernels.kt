package shape.komputation.cuda.kernels

object ForwardKernels {

    fun bias() = KernelInstruction(
        "biasKernel",
        "biasKernel",
        "forward/bias/BiasKernel.cu")

    fun dropoutTraining() = KernelInstruction(
        "dropoutTrainingKernel",
        "dropoutTrainingKernel",
        "forward/dropout/DropoutTrainingKernel.cu",
        listOf(KernelHeaders.zero))

    fun dropoutRuntime() = KernelInstruction(
        "dropoutRuntimeKernel",
        "dropoutRuntimeKernel",
        "forward/dropout/DropoutRuntimeKernel.cu",
        listOf(KernelHeaders.zero))

    fun backwardDropout() = KernelInstruction(
        "backwardDropoutKernel",
        "backwardDropoutKernel",
        "forward/dropout/BackwardDropoutKernel.cu",
        listOf(KernelHeaders.zero))

    fun exponentiation() = KernelInstruction(
        "exponentiationKernel",
        "exponentiationKernel",
        "forward/exponentiation/ExponentiationKernel.cu",
        listOf(KernelHeaders.zero))

    fun backwardExponentiation() = KernelInstruction(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "forward/exponentiation/BackwardExponentiationKernel.cu",
        listOf(KernelHeaders.zero))

    fun normalization(blockSize : Int) = KernelInstruction(
        "normalizationKernel",
        "normalizationKernel<$blockSize>",
        "forward/normalization/NormalizationKernel.cu",
        listOf(KernelHeaders.reduction, KernelHeaders.zero))

    fun backwardNormalization(blockSize: Int) = KernelInstruction(
        "backwardNormalizationKernel",
        "backwardNormalizationKernel<$blockSize>",
        "forward/normalization/BackwardNormalizationKernel.cu",
        listOf(KernelHeaders.reduction, KernelHeaders.zero))

    fun sigmoid() = KernelInstruction(
        "sigmoidKernel",
        "sigmoidKernel",
        "forward/sigmoid/SigmoidKernel.cu",
        listOf(KernelHeaders.zero))

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "backwardSigmoidKernel",
        "forward/sigmoid/BackwardSigmoidKernel.cu",
        listOf(KernelHeaders.zero))

    fun relu() = KernelInstruction(
        "reluKernel",
        "reluKernel",
        "forward/relu/ReluKernel.cu",
        listOf(KernelHeaders.zero))

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "backwardReluKernel",
        "forward/relu/BackwardReluKernel.cu",
        listOf(KernelHeaders.zero))

    fun tanh() = KernelInstruction("tanhKernel",
        "tanhKernel",
        "forward/tanh/TanhKernel.cu",
        listOf(KernelHeaders.zero))

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "backwardTanhKernel",
        "forward/tanh/BackwardTanhKernel.cu",
        listOf(KernelHeaders.zero))
}