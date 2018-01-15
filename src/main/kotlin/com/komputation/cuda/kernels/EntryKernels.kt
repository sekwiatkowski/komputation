package com.komputation.cuda.kernels

object EntryKernels {

    fun lookup() = KernelInstruction(
        "lookupKernel",
        "entry/LookupKernel.cu")

}