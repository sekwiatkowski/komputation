package com.komputation.cuda.kernels

object RecurrentKernels {

    fun kernel() = KernelInstruction(
        "recurrentKernel",
        "recurrentKernel",
        "recurrent/RecurrentKernel.cu")

}