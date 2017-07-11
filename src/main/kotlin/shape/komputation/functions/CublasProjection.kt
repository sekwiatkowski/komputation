package shape.komputation.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.runtime.JCuda.cudaFree
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation.CUBLAS_OP_N
import jcuda.runtime.JCuda.cudaMalloc
import jcuda.jcublas.cublasOperation.CUBLAS_OP_T

fun cublasProject(input: DoubleArray, numberWeightRows : Int, numberWeightColumns: Int, numberWeightEntries: Int, weights : DoubleArray, bias : DoubleArray? = null): DoubleArray {

    val cublasHandle = cublasHandle()
    cublasCreate(cublasHandle)

    val hostResult = DoubleArray(numberWeightRows)

    val deviceWeights = Pointer()
    val deviceInputs = Pointer()
    val deviceResult = Pointer()

    // Allocate memory
    cudaMalloc(deviceWeights,(numberWeightEntries * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceInputs, (numberWeightColumns * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceResult, (numberWeightRows * Sizeof.DOUBLE).toLong())

    // Set the vectors on the device
    cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(weights), 1, deviceWeights, 1)
    cublasSetVector(numberWeightColumns, Sizeof.DOUBLE, Pointer.to(input), 1, deviceInputs, 1)
    cublasSetVector(numberWeightRows, Sizeof.DOUBLE, Pointer.to(bias ?: hostResult), 1, deviceResult, 1)

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

    cublasDestroy(cublasHandle)

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
fun cublasBackwardProjectionWrtInput(hostWeights: DoubleArray, numberWeightRows: Int, numberWeightColumns: Int, numberWeightEntries: Int, hostChain: DoubleArray): DoubleArray {

    val cublasHandle = cublasHandle()
    cublasCreate(cublasHandle)

    val deviceWeights = Pointer()
    val deviceChain = Pointer()
    val deviceResult = Pointer()

    cudaMalloc(deviceWeights, (numberWeightEntries * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceChain, (numberWeightRows * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceResult, (numberWeightColumns * Sizeof.DOUBLE).toLong())

    val hostResult = DoubleArray(numberWeightColumns)

    cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(hostWeights), 1, deviceWeights, 1)
    cublasSetVector(numberWeightRows, Sizeof.DOUBLE, Pointer.to(hostChain), 1, deviceChain, 1)
    cublasSetVector(numberWeightColumns, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

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

    cublasGetVector(numberWeightColumns, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cudaFree(deviceWeights)
    cudaFree(deviceChain)
    cudaFree(deviceResult)

    cublasDestroy(cublasHandle)

    return hostResult

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
fun cublasBackwardProjectionWrtWeights(hostInput: DoubleArray, inputDimension : Int, hostChain: DoubleArray, chainDimension : Int, size : Int): DoubleArray {

    val cublasHandle = cublasHandle()
    cublasCreate(cublasHandle)

    val deviceChain = Pointer()
    val deviceInput = Pointer()
    val deviceResult = Pointer()

    cudaMalloc(deviceChain, (chainDimension * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceInput, (inputDimension * Sizeof.DOUBLE).toLong())
    cudaMalloc(deviceResult, (size * Sizeof.DOUBLE).toLong())

    val hostResult = DoubleArray(size)

    cublasSetVector(chainDimension, Sizeof.DOUBLE, Pointer.to(hostChain), 1, deviceChain, 1)
    cublasSetVector(inputDimension, Sizeof.DOUBLE, Pointer.to(hostInput), 1, deviceInput, 1)
    cublasSetVector(size, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

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

    cublasGetVector(size, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cudaFree(deviceInput)
    cudaFree(deviceChain)
    cudaFree(deviceResult)

    cublasDestroy(cublasHandle)

    return hostResult

}