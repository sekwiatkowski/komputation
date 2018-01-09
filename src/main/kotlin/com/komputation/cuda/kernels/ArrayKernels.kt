package com.komputation.cuda.kernels

object ArrayKernels {

    private val subDirectory = "arrays"

    fun copyBlock() = KernelInstruction(
        "copyBlockKernel",
        "$subDirectory/copy/CopyBlockKernel.cu",
        listOf(KernelHeaders.nan))

    fun replaceNaN() = KernelInstruction(
        "replaceNaNKernel",
        "$subDirectory/nan/ReplaceNaN.cu",
        listOf(KernelHeaders.nan))

    fun add() = KernelInstruction(
        "addKernel",
        "$subDirectory/add/AddKernel.cu")

    fun groupSum() = KernelInstruction(
        "groupSumKernel",
        "$subDirectory/sum/GroupSumKernel.cu")

    fun sum() = KernelInstruction(
        "sumKernel",
        "$subDirectory/sum/SumKernel.cu")

}