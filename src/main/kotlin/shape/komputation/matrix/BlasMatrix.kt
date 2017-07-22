package shape.komputation.matrix

import org.jblas.FloatMatrix
import org.jblas.SimpleBlas

class BlasMatrix(private val matrix: FloatMatrix) {

    fun numberRows () = this.matrix.rows
    fun numberColumns () = this.matrix.columns

    fun getEntries() =

        this.matrix.data!!

    fun multiply(otherMatrix : BlasMatrix): BlasMatrix {

        val result = FloatMatrix(this.numberRows(), otherMatrix.numberColumns())

        SimpleBlas.gemm(1.0f, this.matrix, otherMatrix.matrix, 1.0f, result)

        return BlasMatrix(result)
    }

    fun multiplyAdd(otherMatrix : BlasMatrix, bias : BlasMatrix): BlasMatrix {

        val result = FloatMatrix(this.numberRows(), otherMatrix.numberColumns())

        SimpleBlas.gemm(1.0f, this.matrix, otherMatrix.matrix, 1.0f, result)

        return BlasMatrix(result.add(bias.matrix))

    }

}

fun createBlasMatrix(numberRows: Int, numberColumns: Int, entries : FloatArray) =

    BlasMatrix(FloatMatrix(numberRows, numberColumns, *entries))

fun createBlasVector(entries: FloatArray) =

    createBlasMatrix(entries.size, 1, entries)