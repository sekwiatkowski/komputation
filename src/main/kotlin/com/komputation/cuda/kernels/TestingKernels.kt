package com.komputation.cuda.kernels

object TestingKernels {

    fun binary() = KernelInstruction(
        "binaryTestingKernel",
        "binaryTestingKernel",
        "testing/BinaryTestingKernel.cu")

    fun multiClass() = KernelInstruction(
        "multiClassTestingKernel",
        "multiClassTestingKernel",
        "testing/MultiClassTestingKernel.cu")

}