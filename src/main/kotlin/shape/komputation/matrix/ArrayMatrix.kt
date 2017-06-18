package shape.komputation.matrix

import java.util.*

sealed class Matrix

open class DoubleMatrix(val numberRows : Int, val numberColumns : Int, val entries : DoubleArray) : Matrix()

class SequenceMatrix(val numberSteps : Int, val numberStepRows: Int, val numberStepColumns: Int, entries : DoubleArray) : DoubleMatrix(numberStepRows, numberSteps * numberStepColumns, entries) {

    private val stepSize = numberStepRows * numberStepColumns

    fun setEntry(step: Int, indexRow: Int, indexColumn: Int, entry : Double) {

        entries[step * stepSize + indexColumn * numberStepRows + indexRow] = entry

    }

    fun getStep(step: Int) : DoubleMatrix {

        val start = step * stepSize
        val end = start + stepSize

        return DoubleMatrix(numberStepRows, numberStepColumns, Arrays.copyOfRange(this.entries, start, end))

    }

    fun setStep(step: Int, entries: DoubleArray) {

        System.arraycopy(entries, 0, this.entries, step * stepSize, entries.size)

    }

}

class IntMatrix(val entries : IntArray, val numberRows : Int, val numberColumns : Int) : Matrix()
