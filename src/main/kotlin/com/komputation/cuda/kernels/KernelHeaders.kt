package com.komputation.cuda.kernels

object KernelHeaders {

    val nan = "symbols/NaN.cuh"

    val sumReduction = "reduction/SumReduction.cuh"
    val productReduction = "reduction/ProductReduction.cuh"

    val recurrentActivation = "recurrent/RecurrentActivation.cuh"

    val recurrent = "recurrent/Recurrent.cuh"
    val backwardRecurrent = "recurrent/BackwardRecurrent.cuh"

    val relu = "entrywise/Relu.cuh"
    val sigmoid = "entrywise/Sigmoid.cuh"
    val tanh = "entrywise/Tanh.cuh"

    val addCooperatively = "arrays/add/AddCooperatively.cuh"
    val copyCooperatively = "arrays/copy/CopyCooperatively.cuh"

}