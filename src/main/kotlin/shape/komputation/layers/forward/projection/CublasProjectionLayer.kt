package shape.komputation.layers.forward.projection

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas
import shape.komputation.layers.ForwardLayer
import shape.komputation.matrix.DoubleMatrix
import shape.komputation.optimization.Optimizable
import java.util.*

class CublasProjectionLayer internal constructor(
    name : String?,
    private val weights : DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int) : ForwardLayer(name), Optimizable {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    override fun forward(input: DoubleMatrix, isTraining : Boolean) : DoubleMatrix {

        // Initialize JCublas
        JCublas.cublasInit()

        val numberInputColumns = input.numberColumns
        val numberInputRows = input.numberRows

        val numberResultEntries = numberWeightRows * numberInputColumns
        val numberInputEntries = numberInputRows * numberInputColumns

        /* Allocate host memory for the matrices */
        val hostResult = DoubleArray(numberResultEntries)

        /* Allocate device memory for the matrices */
        val deviceFirst = Pointer()
        val deviceSecond = Pointer()
        val deviceResult = Pointer()

        JCublas.cublasAlloc(this.numberWeightEntries, Sizeof.DOUBLE, deviceFirst)
        JCublas.cublasAlloc(numberInputEntries, Sizeof.DOUBLE, deviceSecond)
        JCublas.cublasAlloc(numberResultEntries, Sizeof.DOUBLE, deviceResult)

        /* Initialize the device matrices with the host matrices */
        JCublas.cublasSetVector(this.numberWeightEntries, Sizeof.DOUBLE, Pointer.to(this.weights), 1, deviceFirst, 1)
        JCublas.cublasSetVector(numberInputEntries, Sizeof.DOUBLE, Pointer.to(input.entries), 1, deviceSecond, 1)
        JCublas.cublasSetVector(numberResultEntries, Sizeof.DOUBLE, Pointer.to(hostResult), 1, deviceResult, 1)

        /* Performs operation using JCublas */
        // C = alpha * op(A) * op(B) + beta * C,
        JCublas.cublasDgemm(
            'n', // no transposition
            'n', // no transposition
            this.numberWeightRows, // number of rows of matrices A and C
            numberInputColumns, // number of columns of matrices B and C
            this.numberWeightColumns, // number of columns of matrix A and number of rows of matrix B
            1.0, // alpha
            deviceFirst, // first pointer
            this.numberWeightRows, // number weight rows
            deviceSecond, // second pointer
            numberInputRows, // number input rows
            0.0, // beta
            deviceResult, // result pointer
            numberWeightRows // number result rows
        )

        /* Read the result back */
        JCublas.cublasGetVector(numberResultEntries, Sizeof.DOUBLE, deviceResult, 1, Pointer.to(hostResult), 1)

        /* Memory clean up */
        JCublas.cublasFree(deviceFirst)
        JCublas.cublasFree(deviceSecond)
        JCublas.cublasFree(deviceResult)

        /* Shutdown */
        JCublas.cublasShutdown()

        return DoubleMatrix(this.numberWeightRows, numberInputColumns, hostResult)

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        TODO()

    }

    override fun optimize(scalingFactor : Double) {

        TODO()

    }

}
