package shape.komputation.matrix

import no.uib.cipr.matrix.DenseMatrix

class BlasMatrix(private val matrix: DenseMatrix) {

    fun numberRows () = matrix.numRows()
    fun numberColumns () = matrix.numColumns()

    fun getEntries() =

        this.matrix.data!!

    fun multiply(otherMatrix : BlasMatrix) =

        BlasMatrix(this.matrix.mult(otherMatrix.matrix, DenseMatrix(this.numberRows(), otherMatrix.numberColumns())) as DenseMatrix)

    fun multiplyAdd(otherMatrix : BlasMatrix, bias : BlasMatrix) =

        BlasMatrix(this.matrix.multAdd(otherMatrix.matrix, bias.matrix) as DenseMatrix)

}

fun createBlasMatrix(numberRows: Int, numberColumns: Int, entries : DoubleArray, deep : Boolean = false) =

    BlasMatrix(DenseMatrix(numberRows, numberColumns, entries, deep))

fun createBlasVector(entries: DoubleArray, deep: Boolean = false) =

    createBlasMatrix(entries.size, 1, entries, deep)