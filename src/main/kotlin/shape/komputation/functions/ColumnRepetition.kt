package shape.komputation.functions

fun repeatColumn(column: DoubleArray, n: Int): DoubleArray {

    val numberRows = column.size

    val repetition = DoubleArray(numberRows * n)

    for(indexColumn in 0..n - 1) {

        val start = indexColumn * numberRows

        for (indexRow in 0..numberRows - 1) {

            repetition[start+indexRow] = column[indexRow]

        }

    }

    return repetition


}