package shape.komputation.cpu.functions

fun sumRows(entries: FloatArray, numberRows: Int, numberColumns : Int) =

    FloatArray(numberRows) { indexRow ->

        var sum = 0.0f

        for (indexColumn in 0..numberColumns - 1) {

            sum += entries[indexColumn * numberRows + indexRow]

        }

        sum

    }