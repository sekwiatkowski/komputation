package shape.konvolution.matrix

import no.uib.cipr.matrix.DenseMatrix

class RealMatrix(private val matrix: DenseMatrix) : Matrix {

    fun numberRows () = matrix.numRows()
    fun numberColumns () = matrix.numColumns()

    fun get(indexRow: Int, indexColumn : Int) =

        this.matrix.get(indexRow, indexColumn)

    fun set(indexRow: Int, indexColumn : Int, value: Double) {

        this.matrix.set(indexRow, indexColumn, value)

    }

    fun add(indexRow: Int, indexColumn : Int, value: Double) {

        this.matrix.add(indexRow, indexColumn, value)

    }

    fun multiply(otherMatrix : RealMatrix) =

        RealMatrix(this.matrix.mult(otherMatrix.matrix, DenseMatrix(this.numberRows(), otherMatrix.numberColumns())) as DenseMatrix)

    fun multiplyAdd(otherMatrix : RealMatrix, bias : RealMatrix) =

        RealMatrix(this.matrix.multAdd(otherMatrix.matrix, bias.matrix) as DenseMatrix)

    fun copy() =

        RealMatrix(this.matrix.copy())


}

fun createRealMatrix(vararg rows: DoubleArray) =

    RealMatrix(DenseMatrix(rows))

fun createRealMatrix(numberRows: Int, numberColumns: Int) =

    RealMatrix(DenseMatrix(numberRows, numberColumns))

fun createRealVector(numberRows: Int) =

    createRealMatrix(numberRows, 1)

fun createOneHotVector(size : Int, oneHotIndex: Int, value : Double = 1.0): RealMatrix {

    val vector = createRealVector(size)

    vector.set(oneHotIndex, 0, value)

    return vector

}