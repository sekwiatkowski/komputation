package shape.komputation.cpu.functions

fun isPaddedColumn(numberRows: Int, indexColumn: Int, input: FloatArray): Boolean {

    for (indexRow in 0..numberRows - 1) {

        val indexEntry = indexColumn * numberRows + indexRow

        val entry = input[indexEntry]

        if (entry.isNaN()) {

            return true

        }

    }

    return false

}


fun findFirstPaddedColumn(input : FloatArray, numberRows : Int, numberColumns : Int): Int {

    for (indexColumn in 0..numberColumns - 1) {

        if (isPaddedColumn(numberRows, indexColumn, input)) {

            return indexColumn

        }

    }

    return -1

}