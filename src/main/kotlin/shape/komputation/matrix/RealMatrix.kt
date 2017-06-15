package shape.komputation.matrix

import no.uib.cipr.matrix.DenseMatrix
import no.uib.cipr.matrix.DenseVector
import java.util.*

class RealMatrix(private val matrix: DenseMatrix) : Matrix {

    fun numberRows () = matrix.numRows()
    fun numberColumns () = matrix.numColumns()

    fun get(indexRow: Int, indexColumn : Int) =

        this.matrix.get(indexRow, indexColumn)

    fun getColumn(indexRow: Int): RealMatrix {

        val from = indexRow * this.numberColumns()
        val to = (indexRow+1) * this.numberColumns()

        val vector = DenseVector(Arrays.copyOfRange(this.matrix.data, from, to), false)

        return RealMatrix(DenseMatrix(vector, false))

    }

    fun getEntries() =

        this.matrix.data

    fun set(indexRow: Int, indexColumn : Int, value: Double) {

        this.matrix.set(indexRow, indexColumn, value)

    }

    fun add(otherMatrix: RealMatrix): RealMatrix {

        val result = this.matrix.add(otherMatrix.matrix) as DenseMatrix

        return RealMatrix(result)

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

    fun maxRow() : Int {

        var maxValue = Double.NEGATIVE_INFINITY
        var maxRow = -1

        for (indexRow in 0..numberRows() - 1) {

            for (indexColumn in 0..numberColumns() - 1) {

                val value = get(indexRow, indexColumn)

                if (value > maxValue) {

                    maxValue = value
                    maxRow = indexRow

                }

            }

        }

        return maxRow

    }

    fun zero() {

        this.matrix.zero()

    }


}

fun createRealMatrix(vararg rows: DoubleArray) =

    RealMatrix(DenseMatrix(rows))

fun createRealMatrix(numberRows: Int, numberColumns: Int) =

    RealMatrix(DenseMatrix(numberRows, numberColumns))

fun createRealMatrix(numberRows: Int, numberColumns: Int, entries : DoubleArray) =

    RealMatrix(DenseMatrix(numberRows, numberColumns, entries, false))

fun createRealVector(numberRows: Int, entries : DoubleArray) =

    RealMatrix(DenseMatrix(numberRows, 1, entries, false))

fun createRealVector(vararg entries : Double) : RealMatrix {

    val numberEntries = entries.size

    val vector = createRealMatrix(numberEntries, 1)

    for (indexEntry in 0..numberEntries - 1) {

        vector.set(indexEntry,0, entries[indexEntry])

    }

    return vector

}

fun createRealVector(numberRows: Int) =

    createRealMatrix(numberRows, 1)

fun createOneHotVector(size : Int, oneHotIndex: Int, value : Double = 1.0): RealMatrix {

    val vector = createRealVector(size)

    vector.set(oneHotIndex, 0, value)

    return vector

}

val EMPTY_MATRIX = createRealMatrix(0, 0)