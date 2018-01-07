package com.komputation.cuda.kernels

object KernelHeaders {

    val nan = "symbols/NaN.cuh"

    val sumReduction = "reduction/SumReduction.cuh"
    val productReduction = "reduction/ProductReduction.cuh"

    val recurrentActivation = "constants/Activation.cuh"
    val resultExtraction = "constants/ResultExtraction.cuh"

    val relu = "entrywise/Relu.cuh"
    val sigmoid = "entrywise/Sigmoid.cuh"
    val tanh = "entrywise/Tanh.cuh"

}