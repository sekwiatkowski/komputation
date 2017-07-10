package shape.komputation.layers.forward.projection

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas.*
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable

class CublasProjectionLayer internal constructor(
    name : String?,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val bias : DoubleArray? = null) : ForwardLayer(name), Optimizable {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

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

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        TODO()

    }

    override fun optimize(scalingFactor : Double) {

        TODO()

    }

}
