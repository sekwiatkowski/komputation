package shape.komputation.layers.forward.projection

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas.*
import shape.komputation.functions.cublasBackwardProjectionWrtInput
import shape.komputation.functions.cublasBackwardProjectionWrtWeights
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

        cublasInit()

        val hostResult = DoubleArray(this.numberWeightRows)

        val deviceWeights = Pointer()
        val deviceInputs = Pointer()
        val deviceResult = Pointer()

        // Allocate memory
        cublasAlloc(this.numberWeightEntries, Sizeof.DOUBLE, deviceWeights)
        cublasAlloc(this.numberWeightColumns, Sizeof.DOUBLE, deviceInputs)
        cublasAlloc(this.numberWeightRows, Sizeof.DOUBLE, deviceResult)

        // Set the vectors on the device
        cublasSetVector(this.numberWeightEntries, Sizeof.DOUBLE, Pointer.to(this.weights), 1, deviceWeights, 1)
        cublasSetVector(this.numberWeightColumns, Sizeof.DOUBLE, Pointer.to(input.entries), 1, deviceInputs, 1)
        cublasSetVector(this.numberWeightRows, Sizeof.DOUBLE, Pointer.to(this.bias ?: hostResult), 1, deviceResult, 1)

        // C = alpha * op(A) * op(B) + beta * C
        val beta = if (this.bias != null) 1.0 else 0.0
        cublasDgemv(
            'n', // no transposition
            this.numberWeightRows, // number of rows of matrix A
            this.numberWeightColumns, // number of columns of matrix A
            1.0, // alpha
            deviceWeights, // weight pointer
            this.numberWeightRows, // number weight rows
            deviceInputs, // input pointer
            1, // storage spacing between elements of x
            beta, // beta
            deviceResult, // result pointer
            this.numberWeightRows // number result rows
        )

        cublasGetVector(this.numberWeightRows, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

        cublasFree(deviceWeights)
        cublasFree(deviceInputs)
        cublasFree(deviceResult)

        cublasShutdown()

        return DoubleMatrix(this.numberWeightRows, 1, hostResult)

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

        val backwardWrtInput = cublasBackwardProjectionWrtInput(this.weights, this.numberWeightRows, this.numberWeightColumns, this.numberWeightEntries, chain.entries)

        val backwardWrtWeights = cublasBackwardProjectionWrtWeights(this.inputEntries, this.inputDimension, chain.entries, this.chainDimension, this.numberWeightEntries)

        this.weightAccumulator.accumulate(backwardWrtWeights)

        if (this.biasAccumulator != null) {

            this.biasAccumulator.accumulate(chain.entries)

        }

        return DoubleMatrix(this.inputDimension, 1, backwardWrtInput)

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