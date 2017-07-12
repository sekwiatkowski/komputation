package shape.komputation.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.functions.cublasBackwardProjectionWrtBias
import shape.komputation.functions.cublasBackwardProjectionWrtInput
import shape.komputation.functions.cublasBackwardProjectionWrtWeights
import shape.komputation.functions.cublasProject
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.initialization.initializeColumnVector
import shape.komputation.initialization.initializeWeights
import shape.komputation.layers.ForwardLayer
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.*
import shape.komputation.optimization.CublasOptimizationStrategy
import shape.komputation.optimization.CublasUpdateRule
import shape.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    private val initialWeights: DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightUpdateRule: CublasUpdateRule? = null,

    private val initialBias: DoubleArray? = null,
    private val biasUpdateRule: CublasUpdateRule? = null) : ForwardLayer(name), Optimizable, Resourceful {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val numberBiasEntries = if(this.initialBias != null) this.initialBias.size else 0

    private val inputDimension = this.numberWeightColumns
    private val resultDimension = this.numberWeightRows
    private val chainDimension = resultDimension

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

    var deviceInput = Pointer()
    var deviceResult = Pointer()
    var deviceChain = Pointer()

    var deviceWeights = Pointer()
    var deviceWeightGradientAccumulator = Pointer()

    var deviceBias = Pointer()
    var deviceBiasGradientAccumulator = Pointer()

    var deviceBackwardWrtInput = Pointer()

    override fun acquire() {

        cublasCreate(this.cublasHandle)

        this.deviceInput = allocateDeviceMemory(this.inputDimension)
        this.deviceResult = allocateDeviceMemory(this.numberWeightRows)
        this.deviceChain = allocateDeviceMemory(this.chainDimension)

        this.deviceBackwardWrtInput = allocateDeviceMemory(this.inputDimension)
        this.deviceWeightGradientAccumulator = allocateDeviceMemory(this.numberWeightEntries)
        this.deviceBiasGradientAccumulator = allocateDeviceMemory(this.numberBiasEntries)

        this.deviceWeights = copyFromHostToDevice(this.initialWeights, this.numberWeightEntries)

        if(this.initialBias != null) {

            this.deviceBias = copyFromHostToDevice(this.initialBias, this.numberBiasEntries)

        }

    }

    override fun forward(input: DoubleMatrix, isTraining: Boolean): DoubleMatrix {

        this.inputEntries = input.entries

        setVector(this.deviceInput, this.inputEntries, this.inputDimension)

        val result = cublasProject(this.cublasHandle, this.deviceInput, this.deviceResult, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, this.deviceBias, this.numberBiasEntries)

        return DoubleMatrix(resultDimension, 1, result)

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

    override fun backward(chain: DoubleMatrix): DoubleMatrix {

        setVector(this.deviceChain, chain.entries, chainDimension)

        cublasBackwardProjectionWrtInput(this.cublasHandle, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, this.deviceChain, this.deviceBackwardWrtInput)
        cublasBackwardProjectionWrtWeights(this.cublasHandle, this.deviceInput, this.deviceChain, this.deviceWeightGradientAccumulator, this.numberWeightRows, this.numberWeightColumns)

        if (this.initialBias != null) {

            cublasBackwardProjectionWrtBias(this.cublasHandle, this.deviceChain, this.chainDimension, this.deviceBiasGradientAccumulator)

        }

        val hostInputResult = getVector(this.deviceBackwardWrtInput, this.inputDimension)

        return DoubleMatrix(this.inputDimension, 1, hostInputResult)

    }

    override fun optimize(scalingFactor: Double) {

        if (this.weightUpdateRule != null) {

            this.weightUpdateRule.update(this.deviceWeights, scalingFactor, this.deviceWeightGradientAccumulator)
            setVectorToZero(this.deviceWeightGradientAccumulator, this.numberWeightEntries)

        }

        if (this.biasUpdateRule != null) {

            this.biasUpdateRule.update(this.deviceBias, scalingFactor, this.deviceBiasGradientAccumulator)
            setVectorToZero(this.deviceBiasGradientAccumulator, this.numberBiasEntries)

        }

    }

    override fun release() {

        cudaFree(this.deviceInput)
        cudaFree(this.deviceResult)
        cudaFree(this.deviceChain)

        cudaFree(this.deviceWeights)
        cudaFree(this.deviceWeightGradientAccumulator)

        if(this.initialBias != null) {

            cudaFree(this.deviceBias)
            cudaFree(this.deviceBiasGradientAccumulator)

        }

        cudaFree(this.deviceBackwardWrtInput)

        cublasDestroy(this.cublasHandle)

    }

}

fun cublasProjectionLayer(
    inputDimension: Int,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : CublasOptimizationStrategy? = null) =

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
    optimizationStrategy : CublasOptimizationStrategy? = null): CublasProjectionLayer {

    val cublasHandle = cublasHandle()

    val numberWeightRows = outputDimension
    val numberWeightColumns = inputDimension

    val weights = initializeWeights(weightInitializationStrategy, numberWeightRows, numberWeightColumns, inputDimension)
    val weightUpdateRule = optimizationStrategy?.invoke(cublasHandle, numberWeightRows, numberWeightColumns)

    val bias : DoubleArray?
    val biasUpdateRule: CublasUpdateRule?

    if (biasInitializationStrategy != null) {

        bias = initializeColumnVector(biasInitializationStrategy, outputDimension)
        biasUpdateRule = optimizationStrategy?.invoke(cublasHandle, bias.size, 1)

    }
    else {

        bias = null
        biasUpdateRule = null

    }

    return CublasProjectionLayer(name, cublasHandle, weights, numberWeightRows, numberWeightColumns, weightUpdateRule, bias, biasUpdateRule)

}