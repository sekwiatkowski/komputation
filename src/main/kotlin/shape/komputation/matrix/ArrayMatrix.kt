package shape.komputation.matrix

import java.util.*

sealed class Matrix(val numberRows: Int, val numberColumns: Int)

open class DoubleMatrix(numberRows: Int, numberColumns: Int, val entries: DoubleArray) : Matrix(numberRows, numberColumns) {

    fun getEntry(indexRow: Int, indexColumn: Int): Double {

        return this.entries[indexColumn * this.numberRows + indexRow]

    }


    fun setEntry(indexRow: Int, indexColumn: Int, entry : Double) {

        this.entries[indexColumn * this.numberRows + indexRow] = entry

    }

    fun getColumn(index: Int) : DoubleMatrix {

        val start = index * this.numberRows
        val end = start + this.numberRows

        return doubleColumnVector(*Arrays.copyOfRange(this.entries, start, end))

    }

    fun setColumn(index: Int, entries: DoubleArray) {

        System.arraycopy(entries, 0, this.entries, index * this.numberRows, entries.size)

    }

}

class IntMatrix(numberRows : Int, numberColumns : Int, val entries : IntArray) : Matrix(numberRows, numberColumns)
