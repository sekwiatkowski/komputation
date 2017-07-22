package shape.komputation.matrix

import java.util.*

sealed class Matrix(val numberRows: Int, val numberColumns: Int)

open class FloatMatrix(numberRows: Int, numberColumns: Int, val entries: FloatArray) : Matrix(numberRows, numberColumns) {

    fun getEntry(indexRow: Int, indexColumn: Int): Float {

        return this.entries[indexColumn * this.numberRows + indexRow]

    }


    fun setEntry(indexRow: Int, indexColumn: Int, entry : Float) {

        this.entries[indexColumn * this.numberRows + indexRow] = entry

    }

    fun getColumn(index: Int) : FloatMatrix {

        val start = index * this.numberRows
        val end = start + this.numberRows

        return floatColumnVector(*Arrays.copyOfRange(this.entries, start, end))

    }

    fun setColumn(index: Int, entries: FloatArray) {

        System.arraycopy(entries, 0, this.entries, index * this.numberRows, entries.size)

    }

}

class IntMatrix(numberRows : Int, numberColumns : Int, val entries : IntArray) : Matrix(numberRows, numberColumns)
