package shape.komputation.matrix

class IntegerMatrix(private val numberColumns: Int, private vararg val values : IntArray) : Matrix {

    fun numberRows () = values.size
    fun numberColumns () = numberColumns

    fun get(indexRow: Int, indexColumn : Int) =

        values[indexRow][indexColumn]

    fun getColumn(indexColumn: Int) =

        IntArray(numberRows()) { indexRow ->

            get(indexRow, indexColumn)
        }

}

fun createIntegerVector(vararg rows: Int) =

    IntegerMatrix(1, *Array(rows.size) { index -> intArrayOf(rows[index]) })