package shape.komputation.layers.forward.projection

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.*
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.*
import jcuda.runtime.cudaStream_t
import shape.komputation.functions.cublasBackwardProjectionWrtInput
import shape.komputation.functions.cublasBackwardProjectionWrtWeights
import shape.komputation.functions.cublasProject
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.*

class CublasProjectionLayer internal constructor(
    name : String?,

    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightAccumulator : DenseAccumulator,
    private val weightUpdateRule: UpdateRule? = null,

    private val bias : DoubleArray? = null,
    private val biasAccumulator: DenseAccumulator? = null,
    private val biasUpdateRule: UpdateRule? = null) : ForwardLayer(name), Optimizable {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val inputDimension = this.numberWeightColumns
    private val chainDimension = this.numberWeightRows

    private var inputEntries = DoubleArray(inputDimension)

    /*
                       i_1
                       i_2
                       i_3
        w_11 w_12 w_13
        w_21 w_22 w_23

        input dimension = number of weight columns
        result dimension = number of weight rows
     */

    override fun forward(input: DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        this.inputEntries = input.entries

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val result = cublasProject(cublasHandle, input.entries, this.numberWeightRows, this.numberWeightColumns, this.numberWeightEntries, this.weights, this.bias)

        cublasDestroy(cublasHandle)

        return DoubleMatrix(this.numberWeightRows, 1, result)

    }

    /*
                          x_1
                          x_2
                          x_3
        w_11 w_12 w_13    w_11 * x_1 + w_12 * x_2 + w_13 * x_3
        w_21 w_22 w_23    w_21 * x_1 + w_22 * x_2 + w_23 * x_3

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

        Differentiation w.r.t weights:

        d Wx / d W = x_1 x_2 x_3
                     x_1 x_2 x_3

        ger solution:
                x1 x2 x3 << transposed x
        chain_1
        chain_2

     */

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val deviceInput = Pointer()
        cudaMalloc(deviceInput, (this.inputDimension * Sizeof.DOUBLE).toLong())
        cublasSetVector(this.inputDimension, Sizeof.DOUBLE, Pointer.to(this.inputEntries), 1, deviceInput, 1)

        val deviceWeights = Pointer()
        cudaMalloc(deviceWeights, (this.numberWeightEntries * Sizeof.DOUBLE).toLong())
        cublasSetVector(this.numberWeightEntries, Sizeof.DOUBLE, Pointer.to(this.weights), 1, deviceWeights, 1)

        val deviceChain = Pointer()
        cudaMalloc(deviceChain, (this.numberWeightRows * Sizeof.DOUBLE).toLong())
        cublasSetVector(this.numberWeightRows, Sizeof.DOUBLE, Pointer.to(chain.entries), 1, deviceChain, 1)

        val deviceInputResult = Pointer()
        cudaMalloc(deviceInputResult, (this.numberWeightColumns * Sizeof.DOUBLE).toLong())
        val hostInputResult = DoubleArray(this.numberWeightColumns)
        cublasSetVector(this.numberWeightColumns, Sizeof.DOUBLE, Pointer.to(hostInputResult), 1, deviceInputResult, 1)

        val deviceWeightResult = Pointer()
        cudaMalloc(deviceWeightResult, (this.numberWeightEntries * Sizeof.DOUBLE).toLong())
        val hostWeightResult = DoubleArray(this.numberWeightEntries)
        cublasSetVector(this.numberWeightEntries, Sizeof.DOUBLE, Pointer.to(hostWeightResult), 1, deviceWeightResult, 1)

        val inputStream = cudaStream_t()
        cudaStreamCreate(inputStream)

        val weightStream = cudaStream_t()
        cudaStreamCreate(weightStream)

        cublasSetStream(cublasHandle, inputStream)
        cublasBackwardProjectionWrtInput(cublasHandle, deviceWeights, this.numberWeightRows, this.numberWeightColumns, deviceChain, deviceInputResult)

        cublasSetStream(cublasHandle, weightStream)
        cublasBackwardProjectionWrtWeights(cublasHandle, deviceInput, deviceChain, this.chainDimension, deviceWeightResult)

        cublasGetVector(this.numberWeightColumns, Sizeof.DOUBLE, deviceInputResult, 1, Pointer.to(hostInputResult), 1)
        cublasGetVector(this.numberWeightEntries, Sizeof.DOUBLE, deviceWeightResult, 1, Pointer.to(hostWeightResult), 1)

        cudaFree(deviceInput)
        cudaFree(deviceWeights)
        cudaFree(deviceChain)

        cudaFree(deviceInputResult)
        cudaFree(deviceWeightResult)

        cudaStreamDestroy(inputStream)
        cudaStreamDestroy(weightStream)

        cublasDestroy(cublasHandle)

        this.weightAccumulator.accumulate(hostWeightResult)

        if (this.biasAccumulator != null) {

            this.biasAccumulator.accumulate(chain.entries)

        }

        return DoubleMatrix(this.inputDimension, 1, hostInputResult)

    }

    override fun optimize(scalingFactor : Double) {

        if (this.weightUpdateRule != null) {

            val weightAccumulator = this.weightAccumulator

            updateDensely(this.weights, weightAccumulator.getAccumulation(), scalingFactor, this.weightUpdateRule)

            weightAccumulator.reset()

        }

        if (this.bias != null && this.biasUpdateRule != null) {

            val biasAccumulator = this.biasAccumulator!!

            updateDensely(this.bias, biasAccumulator.getAccumulation(), scalingFactor, this.biasUpdateRule)

            biasAccumulator.reset()

        }

    }

}

fun cublasProjectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null) =

    cublasProjectionLayer(
        null,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy
    )


fun cublasProjectionLayer(
    name : String?,
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationStrategy? = null): CublasProjectionLayer {

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weights = initializeWeights(weightInitializationStrategy, numberWeightRows, numberWeightColumns, inputDimension)
    val weightUpdateRule = optimizationStrategy?.invoke(numberWeightRows, numberWeightColumns)

    val bias : DoubleArray?
    val biasUpdateRule: UpdateRule?
    val biasAccumulator: DenseAccumulator?

    if (biasInitializationStrategy != null) {

        bias = initializeColumnVector(biasInitializationStrategy, outputDimension)
        biasUpdateRule = optimizationStrategy?.invoke(bias.size, 1)
        biasAccumulator = DenseAccumulator(bias.size)

    }
    else {

        bias = null
        biasUpdateRule = null
        biasAccumulator = null

    }

    val weightAccumulator = DenseAccumulator(numberWeightRows * numberWeightColumns)

    return CublasProjectionLayer(name, weights, numberWeightRows, numberWeightColumns, weightAccumulator, weightUpdateRule, bias, biasAccumulator, biasUpdateRule)

}