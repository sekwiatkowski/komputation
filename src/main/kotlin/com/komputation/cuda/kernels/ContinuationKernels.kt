package com.komputation.cuda.kernels

object ContinuationKernels {

    private val subDirectory = "continuation"

    fun bias() = KernelInstruction(
        "biasKernel",
        "$subDirectory/bias/BiasKernel.cu")

    fun backwardBias() = KernelInstruction(
        "backwardBiasKernel",
        "$subDirectory/bias/BackwardBiasKernel.cu")

    fun dropoutTraining() = KernelInstruction(
        "dropoutTrainingKernel",
        "$subDirectory/dropout/DropoutTrainingKernel.cu")

    fun dropoutRuntime() = KernelInstruction(
        "dropoutRuntimeKernel",
        "$subDirectory/dropout/DropoutRuntimeKernel.cu")

    fun backwardDropout() = KernelInstruction(
        "backwardDropoutKernel",
        "$subDirectory/dropout/BackwardDropoutKernel.cu")

    fun exponentiation() = KernelInstruction(
        "exponentiationKernel",
        "$subDirectory/exponentiation/ExponentiationKernel.cu")

    fun backwardExponentiation() = KernelInstruction(
        "backwardExponentiationKernel",
        "$subDirectory/exponentiation/BackwardExponentiationKernel.cu")

    fun normalization() = KernelInstruction(
        "normalizationKernel",
        "$subDirectory/normalization/NormalizationKernel.cu")

    fun backwardNormalization() = KernelInstruction(
        "backwardNormalizationKernel",
        "$subDirectory/normalization/BackwardNormalizationKernel.cu")

    fun sigmoid() = KernelInstruction(
        "sigmoidKernel",
        "$subDirectory/sigmoid/SigmoidKernel.cu")

    fun backwardSigmoid() = KernelInstruction(
        "backwardSigmoidKernel",
        "$subDirectory/sigmoid/BackwardSigmoidKernel.cu")

    fun relu() = KernelInstruction(
        "reluKernel",
        "$subDirectory/relu/ReluKernel.cu")

    fun backwardRelu() = KernelInstruction(
        "backwardReluKernel",
        "$subDirectory/relu/BackwardReluKernel.cu")

    fun tanh() = KernelInstruction(
        "tanhKernel",
        "$subDirectory/tanh/TanhKernel.cu")

    fun backwardTanh() = KernelInstruction(
        "backwardTanhKernel",
        "$subDirectory/tanh/BackwardTanhKernel.cu")

    fun maxPooling() = KernelInstruction(
        "maxPoolingKernel",
        "$subDirectory/maxpooling/MaxPoolingKernel.cu")

    fun backwardMaxPooling() = KernelInstruction(
        "backwardMaxPoolingKernel",
        "$subDirectory/maxpooling/BackwardMaxPoolingKernel.cu")

    fun expansion() = KernelInstruction(
        "expansionKernel",
        "$subDirectory/expansion/ExpansionKernel.cu")

    fun backwardExpansion() = KernelInstruction(
        "backwardExpansionKernel",
        "$subDirectory/expansion/BackwardExpansionKernel.cu")

    fun recurrentEachStep() = KernelInstruction(
        "recurrentEachStepKernel",
        "$subDirectory/recurrent/eachstep/RecurrentEachStepKernel.cu")

    fun backwardRecurrentEachStep() = KernelInstruction(
        "backwardRecurrentEachStepKernel",
        "$subDirectory/recurrent/eachstep/BackwardRecurrentEachStepKernel.cu")

    fun recurrentLastStep() = KernelInstruction(
        "recurrentLastStepKernel",
        "$subDirectory/recurrent/laststep/RecurrentLastStepKernel.cu")

    fun backwardRecurrentLastStep() = KernelInstruction(
        "backwardRecurrentLastStepKernel",
        "$subDirectory/recurrent/laststep/BackwardRecurrentLastStepKernel.cu")

}