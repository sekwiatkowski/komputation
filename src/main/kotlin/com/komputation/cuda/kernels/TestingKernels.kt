package com.komputation.cuda.kernels

object TestingKernels {

    fun binary() = KernelInstruction(
        "binaryTestingKernel",
        "testing/BinaryTestingKernel.cu")

    fun multiClass() = KernelInstruction(
        "multiClassTestingKernel",
        "testing/MultiClassTestingKernel.cu")

}