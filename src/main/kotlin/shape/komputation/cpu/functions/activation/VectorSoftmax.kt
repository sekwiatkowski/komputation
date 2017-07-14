package shape.komputation.cpu.functions.activation

fun vectorSoftmax(input: DoubleArray) : DoubleArray {

    val results = DoubleArray(input.size)

    var sum = 0.0

    for (index in 0..input.size - 1) {

        val exponentiation = Math.exp(input[index])

        sum += exponentiation

        results[index] = exponentiation

    }

    for (index in 0..input.size - 1) {

        results[index] /= sum
    }

    return results

}


fun backwardVectorSoftmax(forwardEntries: DoubleArray, chainEntries: DoubleArray): DoubleArray {

    val numberEntries = forwardEntries.size

    val gradient = DoubleArray(numberEntries)

    for (outerIndexRow in 0..numberEntries - 1) {

        val forwardEntry = forwardEntries[outerIndexRow]

        var derivative = 0.0

        for (innerIndexRow in 0..numberEntries - 1) {

            val chainEntry = chainEntries[innerIndexRow]

            // i == j
            if (outerIndexRow == innerIndexRow) {

                derivative += chainEntry * forwardEntry * (1 - forwardEntry)

            }
            // i != j
            else {

                val otherForwardEntry = forwardEntries[innerIndexRow]

                derivative += chainEntry * (-forwardEntry * otherForwardEntry)

            }

        }

        gradient[outerIndexRow] = derivative

    }

    return gradient
}
