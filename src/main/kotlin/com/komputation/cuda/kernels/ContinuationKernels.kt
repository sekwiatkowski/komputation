package com.komputation.cuda.kernels

object ContinuationKernels {

    private val subDirectory = "continuation"

    fun bias() = KernelInstruction(
        "biasKernel",
        "biasKernel",
        "$subDirectory/bias/BiasKernel.cu",
        listOf(KernelHeaders.nan))

    fun dropoutTraining() = KernelInstruction(
        "dropoutTrainingKernel",
        "dropoutTrainingKernel",
        "$subDirectory/dropout/DropoutTrainingKernel.cu",
        listOf(KernelHeaders.nan))

    fun dropoutRuntime() = KernelInstruction(
        "dropoutRuntimeKernel",
        "dropoutRuntimeKernel",
        "$subDirectory/dropout/DropoutRuntimeKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardDropout() = KernelInstruction(
        "backwardDropoutKernel",
        "backwardDropoutKernel",
        "$subDirectory/dropout/BackwardDropoutKernel.cu",
        listOf(KernelHeaders.nan))

    fun exponentiation() = KernelInstruction(
        "exponentiationKernel",
        "exponentiationKernel",
        "$subDirectory/exponentiation/ExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardExponentiation() = KernelInstruction(
        "backwardExponentiationKernel",
        "backwardExponentiationKernel",
        "$subDirectory/exponentiation/BackwardExponentiationKernel.cu",
        listOf(KernelHeaders.nan))

    fun normalization() = KernelInstruction(
        "normalizationKernel",
        "normalizationKernel",
        "$subDirectory/normalization/NormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun backwardNormalization() = KernelInstruction(
        "backwardNormalizationKernel",
        "backwardNormalizationKernel",
        "$subDirectory/normalization/BackwardNormalizationKernel.cu",
        listOf(KernelHeaders.sumReduction, KernelHeaders.nan))

    fun sigmoid() = KernelInstruction(
        "sigmoidKernel",
        "sigmoidKernel",
        "$subDirectory/sigmoid/SigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "backwardSigmoidKernel",
        "$subDirectory/sigmoid/BackwardSigmoidKernel.cu",
        listOf(KernelHeaders.nan))

    fun relu() = KernelInstruction(
        "reluKernel",
        "reluKernel",
        "$subDirectory/relu/ReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "backwardReluKernel",
        "$subDirectory/relu/BackwardReluKernel.cu",
        listOf(KernelHeaders.nan))

    fun tanh() = KernelInstruction("tanhKernel",
        "tanhKernel",
        "$subDirectory/tanh/TanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "backwardTanhKernel",
        "$subDirectory/tanh/BackwardTanhKernel.cu",
        listOf(KernelHeaders.nan))

    fun maxPooling() = KernelInstruction(
        "maxPoolingKernel",
        "maxPoolingKernel",
        "$subDirectory/maxpooling/MaxPoolingKernel.cu",
        listOf(KernelHeaders.nan))

    fun backwardMaxPooling() = KernelInstruction(
        "backwardMaxPoolingKernel",
        "backwardMaxPoolingKernel",
        "$subDirectory/maxpooling/BackwardMaxPoolingKernel.cu")

    fun expansion() = KernelInstruction(
        "expansionKernel",
        "expansionKernel",
        "$subDirectory/expansion/ExpansionKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.zero))

    fun backwardExpansion() = KernelInstruction(
        "backwardExpansionKernel",
        "backwardExpansionKernel",
        "$subDirectory/expansion/BackwardExpansionKernel.cu",
        listOf(KernelHeaders.sumReduction))

    fun stack() = KernelInstruction(
        "stackKernel",
        "stackKernel",
        "$subDirectory/stack/StackKernel.cu")

    fun backwardStack() = KernelInstruction(
        "backwardStackKernel",
        "backwardStackKernel",
        "$subDirectory/stack/BackwardStackKernel.cu")

}