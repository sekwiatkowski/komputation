package com.komputation.cuda.kernels

object ArrayKernels {

    private val subDirectory = "arrays"

    fun fillOneIntegerArray() = KernelInstruction(
        "fillOneIntegerArrayKernel",
        "$subDirectory/fill/FillOneIntegerArrayKernel.cu")

    fun fillTwoIntegerArrays() = KernelInstruction(
        "fillTwoIntegerArraysKernel",
        "$subDirectory/fill/FillTwoIntegerArraysKernel.cu")

    fun fillOneFloatArray() = KernelInstruction(
        "fillOneFloatArrayKernel",
        "$subDirectory/fill/FillOneFloatArrayKernel.cu")

    fun fillTwoFloatArrays() = KernelInstruction(
        "fillTwoFloatArraysKernel",
        "$subDirectory/fill/FillTwoFloatArraysKernel.cu")

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
        "$subDirectory/groupsum/GroupSumKernel.cu")

}