package shape.komputation.functions

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas.*

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

    cublasInit()

    val deviceWeights = Pointer()
    val deviceChain = Pointer()
    val deviceResult = Pointer()

    cublasAlloc(numberWeightEntries, Sizeof.DOUBLE, deviceWeights)
    cublasAlloc(numberWeightRows, Sizeof.DOUBLE, deviceChain)
    cublasAlloc(numberWeightColumns, Sizeof.DOUBLE, deviceResult)

    val hostResult = DoubleArray(numberWeightColumns)

    cublasSetVector(numberWeightEntries, Sizeof.DOUBLE, Pointer.to(hostWeights), 1, deviceWeights, 1)
    cublasSetVector(numberWeightRows, Sizeof.DOUBLE, Pointer.to(hostChain), 1, deviceChain, 1)
    cublasSetVector(numberWeightColumns, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

    cublasDgemv(
        't', // transpose
        numberWeightRows, // number of rows of matrix A
        numberWeightColumns, // number of columns of matrix A
        1.0, // alpha
        deviceWeights, // weight pointer
        numberWeightRows, // number weight rows
        deviceChain, // input pointer
        1, // storage spacing between elements of x
        0.0, // beta
        deviceResult, // result pointer
        1 // specifies the storage spacing between elements of y
    )

    cublasGetVector(numberWeightColumns, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cublasFree(deviceWeights)
    cublasFree(deviceChain)
    cublasFree(deviceResult)

    cublasShutdown()

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

    cublasInit()

    val deviceChain = Pointer()
    val deviceInput = Pointer()
    val deviceResult = Pointer()

    cublasAlloc(chainDimension, Sizeof.DOUBLE, deviceChain)
    cublasAlloc(inputDimension, Sizeof.DOUBLE, deviceInput)
    cublasAlloc(size, Sizeof.DOUBLE, deviceResult)

    val hostResult = DoubleArray(size)

    cublasSetVector(chainDimension, Sizeof.DOUBLE, Pointer.to(hostChain), 1, deviceChain, 1)
    cublasSetVector(inputDimension, Sizeof.DOUBLE, Pointer.to(hostInput), 1, deviceInput, 1)
    cublasSetVector(size, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

    cublasDger(
        chainDimension, // rows of matrix A
        chainDimension, // columns of matrix A
        1.0, // alpha
        deviceChain,
        1,
        deviceInput,
        1,
        deviceResult,
        chainDimension // rows of matrix A
    )

    cublasGetVector(size, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

    cublasFree(deviceInput)
    cublasFree(deviceChain)
    cublasFree(deviceResult)

    cublasShutdown()

    return hostResult

}