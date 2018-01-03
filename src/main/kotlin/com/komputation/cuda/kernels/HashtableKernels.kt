package com.komputation.cuda.kernels

object HashtableKernels {

    fun groupSum() = KernelInstruction(
        "groupSumKernel",
        "hashtable/GroupSumKernel.cu")

    fun hash() = KernelInstruction(
        "hashKernel",
        "hashtable/HashKernel.cu")

}