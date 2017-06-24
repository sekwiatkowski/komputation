package shape.komputation.functions

fun sumRows(entries: DoubleArray, numberRows: Int, numberColumns : Int) =

    DoubleArray(numberRows) { indexRow ->

        var sum = 0.0
        for (indexColumn in 0..numberColumns - 1) {

            sum += entries[indexColumn * numberRows + indexRow]

        }

        sum

    }