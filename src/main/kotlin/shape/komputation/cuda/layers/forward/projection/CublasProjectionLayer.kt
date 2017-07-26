package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.functions.*
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setVectorToZero
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name: String?,
    private val cublasHandle: cublasHandle,
    private val initialWeights: FloatArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightUpdateRule: CudaUpdateRule? = null,

    private val initialBias: FloatArray? = null,
    private val biasUpdateRule: CudaUpdateRule? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val hasBias = initialBias != null
    private val numberBiasEntries = if(this.hasBias) this.initialBias!!.size else 0

    private val inputDimension = this.numberWeightColumns
    private val resultDimension = this.numberWeightRows
    private val chainDimension = resultDimension

    /*
                       i_1
                       i_2
                       i_3
        w_11 w_12 w_13
        w_21 w_22 w_23
        input dimension = number of weight columns
        result dimension = number of weight rows
    */

    var deviceResult = Pointer()

    var deviceWeights = Pointer()
    var deviceWeightGradientAccumulator = Pointer()

    var deviceBias = Pointer()
    var deviceBiasGradientAccumulator = Pointer()

    var deviceBackwardWrtInput = Pointer()

    override fun acquire() {

        allocateDeviceFloatMemory(this.deviceResult, this.numberWeightRows)

        allocateDeviceFloatMemory(this.deviceBackwardWrtInput, this.inputDimension)
        allocateDeviceFloatMemory(this.deviceWeightGradientAccumulator, this.numberWeightEntries)
        allocateDeviceFloatMemory(this.deviceBiasGradientAccumulator, this.numberBiasEntries)

        setFloatArray(this.initialWeights, this.numberWeightEntries, this.deviceWeights)

        if(this.hasBias) {

            setFloatArray(this.initialBias!!, this.numberBiasEntries, this.deviceBias)

        }

    }

    private var deviceInput = Pointer()

    override fun forward(input : Pointer, isTraining : Boolean): Pointer {

        this.deviceInput = input

        if (this.hasBias) {

            cublasProjectWithBias(this.cublasHandle, this.deviceInput, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, this.deviceResult, this.numberBiasEntries, this.deviceBias)

        }
        else {

            cublasProject(this.cublasHandle, this.deviceInput, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, this.deviceResult)

        }

        return this.deviceResult

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

    override fun backward(chain: Pointer): Pointer {

        cublasBackwardProjectionWrtInput(this.cublasHandle, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, chain, this.deviceBackwardWrtInput)
        cublasBackwardProjectionWrtWeights(this.cublasHandle, this.deviceInput, chain, this.deviceWeightGradientAccumulator, this.numberWeightRows, this.numberWeightColumns)

        if (this.initialBias != null) {

            cublasBackwardProjectionWrtBias(this.cublasHandle, chain, this.chainDimension, this.deviceBiasGradientAccumulator)

        }

        return this.deviceBackwardWrtInput

    }

    override fun optimize(scalingFactor: Float) {

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

        cudaFree(this.deviceResult)

        cudaFree(this.deviceWeights)
        cudaFree(this.deviceWeightGradientAccumulator)

        if(this.initialBias != null) {

            cudaFree(this.deviceBias)
            cudaFree(this.deviceBiasGradientAccumulator)

        }

        cudaFree(this.deviceBackwardWrtInput)

    }

}