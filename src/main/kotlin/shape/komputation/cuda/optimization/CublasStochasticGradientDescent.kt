package shape.komputation.cuda.optimization

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasDgeam
import jcuda.jcublas.cublasHandle
import jcuda.jcublas.cublasOperation.CUBLAS_OP_N

class CublasStochasticGradientDescent(private val handle : cublasHandle, private val numberRows : Int, private val numberColumns : Int, private val learningRate: Double) : CublasUpdateRule {

    override fun update(deviceParameter: Pointer, scalingFactor : Double, deviceGradient: Pointer) {

        cublasDgeam(
            handle,
            CUBLAS_OP_N, // operation on A
            CUBLAS_OP_N, // operation on B
            numberRows,
            numberColumns,
            Pointer.to(doubleArrayOf(1.0)),
            deviceParameter,
            numberRows,
            Pointer.to(doubleArrayOf(-scalingFactor * this.learningRate)),
            deviceGradient,
            numberRows,
            deviceParameter,
            numberRows)

    }

}