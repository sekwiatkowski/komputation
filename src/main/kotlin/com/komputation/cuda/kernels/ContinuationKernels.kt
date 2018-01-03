package com.komputation.cuda.kernels

object ContinuationKernels {

    private val subDirectory = "continuation"

    fun bias() = KernelInstruction(
        "biasKernel",
        "$subDirectory/bias/BiasKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardBias() = KernelInstruction(
        "backwardBiasKernel",
        "$subDirectory/bias/BackwardBiasKernel.cu",
        listOf(KernelHeaders.sumReduction))

    fun dropoutTraining() = KernelInstruction(
        "dropoutTrainingKernel",
        "$subDirectory/dropout/DropoutTrainingKernel.cu",
        listOf(KernelHeaders.nan))

    fun dropoutRuntime() = KernelInstruction(
        "dropoutRuntimeKernel",
        "$subDirectory/dropout/DropoutRuntimeKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardDropout() = KernelInstruction(
        "backwardDropoutKernel",
        "$subDirectory/dropout/BackwardDropoutKernel.cu",
        listOf(KernelHeaders.nan))

    fun exponentiation() = KernelInstruction(
        "exponentiationKernel",
        "$subDirectory/exponentiation/ExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardExponentiation() = KernelInstruction(
        "backwardExponentiationKernel",
        "$subDirectory/exponentiation/BackwardExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun normalization() = KernelInstruction(
        "normalizationKernel",
        "$subDirectory/normalization/NormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun backwardNormalization() = KernelInstruction(
        "backwardNormalizationKernel",
        "$subDirectory/normalization/BackwardNormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun sigmoid() = KernelInstruction(
        "sigmoidKernel",
        "$subDirectory/sigmoid/SigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "$subDirectory/sigmoid/BackwardSigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun relu() = KernelInstruction(
        "reluKernel",
        "$subDirectory/relu/ReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "$subDirectory/relu/BackwardReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun tanh() = KernelInstruction(
        "tanhKernel",
        "$subDirectory/tanh/TanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "$subDirectory/tanh/BackwardTanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun maxPooling() = KernelInstruction(
        "maxPoolingKernel",
        "$subDirectory/maxpooling/MaxPoolingKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardMaxPooling() = KernelInstruction(
        "backwardMaxPoolingKernel",
        "$subDirectory/maxpooling/BackwardMaxPoolingKernel.cu")

    fun expansion() = KernelInstruction(
        "expansionKernel",
        "$subDirectory/expansion/ExpansionKernel.cu")

    fun backwardExpansion() = KernelInstruction(
        "backwardExpansionKernel",
        "$subDirectory/expansion/BackwardExpansionKernel.cu",
        listOf(KernelHeaders.sumReduction))

}