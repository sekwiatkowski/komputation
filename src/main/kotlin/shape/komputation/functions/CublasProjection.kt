package shape.komputation.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.runtime.JCuda.cudaFree
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation.CUBLAS_OP_N
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T
import shape.komputation.matrix.copyFromHostToDevice

fun cublasProject(cublasHandle: cublasHandle, input: DoubleArray, inputDimension : Int, numberWeightRows : Int, numberWeightColumns: Int, numberWeightEntries: Int, weights : DoubleArray, bias : DoubleArray? = null): DoubleArray {

    val deviceWeights = copyFromHostToDevice(weights, numberWeightEntries)

    val deviceInputs = copyFromHostToDevice(input, inputDimension)

    val hostResult = DoubleArray(numberWeightRows)
    val deviceResult = copyFromHostToDevice(bias ?: hostResult, numberWeightRows)

    // C = alpha * op(A) * op(B) + beta * C
    val beta = if (bias != null) 1.0 else 0.0
    cublasDgemv(
        cublasHandle,
        CUBLAS_OP_N, // no transposition
        numberWeightRows, // number of rows of matrix A
        numberWeightColumns, // number of columns of matrix A
        Pointer.to(doubleArrayOf(1.0)), // alpha
        deviceWeights, // weight pointer
        numberWeightRows, // number weight rows
        deviceInputs, // input pointer
        1, // storage spacing between elements of x
        Pointer.to(doubleArrayOf(beta)), // beta
        deviceResult, // result pointer
        numberWeightRows // number result rows
    )

    cublasGetVector(numberWeightRows, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cudaFree(deviceWeights)
    cudaFree(deviceInputs)
    cudaFree(deviceResult)

    return hostResult

}

/*
    Differentiation w.r.t input:

    d Wx / d x = w_11 + w_21
                 w_12 + w_22
                 w_13 + w_23

    gemv solution:
                              chain_1
                              chain_2
    transposed W >> w_11 w_21
                    w_12 w_22
                    w_13 w_23
 */
fun cublasBackwardProjectionWrtInput(cublasHandle: cublasHandle, deviceWeights: Pointer, numberWeightRows: Int, numberWeightColumns: Int, deviceChain: Pointer, deviceResult : Pointer) {

    cublasDgemv(
        cublasHandle,
        CUBLAS_OP_T, // transpose
        numberWeightRows, // number of rows of matrix A
        numberWeightColumns, // number of columns of matrix A
        Pointer.to(doubleArrayOf(1.0)), // alpha
        deviceWeights, // weight pointer
        numberWeightRows, // number weight rows
        deviceChain, // input pointer
        1, // storage spacing between elements of x
        Pointer.to(doubleArrayOf(0.0)), // beta
        deviceResult, // result pointer
        1 // specifies the storage spacing between elements of y
    )

}

/*
    Differentiation w.r.t weights:

    d Wx / d W = x_1 x_2 x_3
                 x_1 x_2 x_3

    ger solution:
            x1 x2 x3 << transposed x
    chain_1
    chain_2
 */
fun cublasBackwardProjectionWrtWeights(cublasHandle: cublasHandle, deviceInput: Pointer, deviceChain: Pointer, chainDimension : Int, deviceResult: Pointer) {

    cublasDger(
        cublasHandle,
        chainDimension, // rows of matrix A
        chainDimension, // columns of matrix A
        Pointer.to(doubleArrayOf(1.0)), // alpha
        deviceChain,
        1,
        deviceInput,
        1,
        deviceResult,
        chainDimension // rows of matrix A
    )

}