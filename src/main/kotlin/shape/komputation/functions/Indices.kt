package shape.komputation.functions

fun findMaxIndicesInRows(input: DoubleArray, numberRows : Int, numberColumns : Int) =

    IntArray(numberRows) { indexRow ->

        var maxValue = input[indexRow]
        var maxIndex = indexRow

        for (indexColumn in 1..numberColumns - 1) {

            val index = indexRow + indexColumn * numberRows

            val entry = input[index]

            if (entry > maxValue) {

                maxValue = entry
                maxIndex = index

            }

        }

        maxIndex

    }

fun findMaxIndicesInColumns(input: DoubleArray, numberRows : Int, numberColumns : Int) =

    IntArray(numberColumns) { indexColumn ->

        val startIndex = indexColumn * numberRows

        var maxValue = input[startIndex]
        var maxIndex = startIndex

        for (indexRow in 1..numberRows - 1) {

            val index = startIndex + indexRow

            val entry = input[index]

            if (entry > maxValue) {

                maxValue = entry
                maxIndex = index

            }

        }

        maxIndex

    }

fun selectEntries(input: DoubleArray, indices : IntArray) =

    DoubleArray(indices.size) { index -> input[indices[index]] }