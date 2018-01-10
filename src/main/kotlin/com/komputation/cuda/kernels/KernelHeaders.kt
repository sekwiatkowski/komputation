package com.komputation.cuda.kernels

object KernelHeaders {

    val nan = "symbols/NaN.cuh"

    val sumReduction = "reduction/SumReduction.cuh"
    val productReduction = "reduction/ProductReduction.cuh"

    val recurrent = "continuation/recurrent/Recurrent.cuh"
    val backwardRecurrent = "continuation/recurrent/BackwardRecurrent.cuh"
    val recurrentActivation = "continuation/recurrent/RecurrentActivation.cuh"

    val relu = "continuation/relu/Relu.cuh"
    val sigmoid = "continuation/sigmoid/Sigmoid.cuh"
    val tanh = "continuation/tanh/Tanh.cuh"

    val addCooperatively = "arrays/add/AddCooperatively.cuh"
    val copyCooperatively = "arrays/copy/CopyCooperatively.cuh"

}