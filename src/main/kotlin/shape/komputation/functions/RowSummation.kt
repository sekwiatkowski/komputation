package shape.komputation.functions

fun add(a: DoubleArray, b: DoubleArray) =

    DoubleArray(a.size) { index ->

        a[index] + b[index]

    }

fun sumRows(entries: DoubleArray, numberRows: Int, numberColumns : Int): DoubleArray {

    val result = DoubleArray(numberRows)

    for (indexColumn in 0..numberColumns - 1) {

        val start = indexColumn * numberRows

        for (indexRow in 0..numberRows - 1) {

            result[indexColumn] = entries[start + indexRow]

        }

    }

    return result

}