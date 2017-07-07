package shape.komputation.functions

fun findMaxIndicesInRows(input: DoubleArray, numberRows : Int, numberColumns : Int) =

    IntArray(numberRows) { indexRow ->

        var maxValue = input[indexRow]
        var maxIndex = indexRow

        // Go through the columns in the given row
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

fun findMaxIndex(input: DoubleArray): Int {

    var maxIndex = -1
    var maxValue = Double.NEGATIVE_INFINITY

    for(index in 0..input.size - 1) {

        val value = input[index]

        if (value > maxValue) {

            maxValue = value
            maxIndex = index

        }

    }

    return maxIndex

}