package shape.konvolution

import no.uib.cipr.matrix.DenseMatrix

interface Matrix

class RealMatrix : Matrix {

    private val matrix : DenseMatrix

    fun numberRows () = matrix.numRows()
    fun numberColumns () = matrix.numColumns()

    constructor(numberRows : Int, numberColumns : Int) {

        this.matrix = DenseMatrix(numberRows, numberColumns)
    }

    constructor(vararg rows : DoubleArray) {

        this.matrix = DenseMatrix(rows)
    }

    constructor(matrix : DenseMatrix) {

        this.matrix = matrix
    }

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

    fun get(indexRow: Int, indexColumn : Int) =

        this.matrix.get(indexRow, indexColumn)

}

class IntegerMatrix(rows : Array<IntArray>) : Matrix