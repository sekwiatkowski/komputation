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
        listOf(KernelHeaders.nan, KernelHeaders.sigmoid))

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "$subDirectory/sigmoid/BackwardSigmoidKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.sigmoid))

    fun relu() = KernelInstruction(
        "reluKernel",
        "$subDirectory/relu/ReluKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.relu))

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "$subDirectory/relu/BackwardReluKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.relu))

    fun tanh() = KernelInstruction(
        "tanhKernel",
        "$subDirectory/tanh/TanhKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.tanh))

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "$subDirectory/tanh/BackwardTanhKernel.cu",
        listOf(KernelHeaders.nan, KernelHeaders.tanh))

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

    fun recurrentEachStep() = KernelInstruction(
        "recurrentEachStepKernel",
        "$subDirectory/recurrent/eachstep/RecurrentEachStepKernel.cu",
        listOf(KernelHeaders.recurrent, KernelHeaders.recurrentActivation, KernelHeaders.relu, KernelHeaders.sigmoid, KernelHeaders.tanh, KernelHeaders.addCooperatively, KernelHeaders.copyCooperatively, KernelHeaders.nan))

    fun backwardRecurrentEachStep() = KernelInstruction(
        "backwardRecurrentEachStepKernel",
        "$subDirectory/recurrent/eachstep/BackwardRecurrentEachStepKernel.cu",
        listOf(KernelHeaders.backwardRecurrent, KernelHeaders.recurrentActivation, KernelHeaders.relu, KernelHeaders.sigmoid, KernelHeaders.tanh, KernelHeaders.addCooperatively, KernelHeaders.copyCooperatively))

    fun recurrentLastStep() = KernelInstruction(
        "recurrentLastStepKernel",
        "$subDirectory/recurrent/laststep/RecurrentLastStepKernel.cu",
        listOf(KernelHeaders.recurrent, KernelHeaders.recurrentActivation, KernelHeaders.relu, KernelHeaders.sigmoid, KernelHeaders.tanh, KernelHeaders.addCooperatively, KernelHeaders.copyCooperatively, KernelHeaders.nan))

    fun backwardRecurrentLastStep() = KernelInstruction(
        "backwardRecurrentLastStepKernel",
        "$subDirectory/recurrent/laststep/BackwardRecurrentLastStepKernel.cu",
        listOf(KernelHeaders.backwardRecurrent, KernelHeaders.recurrentActivation, KernelHeaders.relu, KernelHeaders.sigmoid, KernelHeaders.tanh, KernelHeaders.addCooperatively, KernelHeaders.copyCooperatively))

}