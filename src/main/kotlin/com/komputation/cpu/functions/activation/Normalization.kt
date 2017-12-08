package com.komputation.cpu.functions.activation

fun normalize(numberRows: Int, numberColumns: Int, input: FloatArray, sums: FloatArray, result: FloatArray) {
    for (indexColumn in 0 until numberColumns) {

        val start = indexColumn * numberRows

        var sum = 0.0f
        for (indexRow in 0 until numberRows) {

            val indexEntry = start + indexRow

            val inputEntry = input[indexEntry]

            sum += inputEntry
            result[indexEntry] = inputEntry

        }

        sums[indexColumn] = sum

        for (indexRow in 0 until numberRows) {

            result[start + indexRow] /= sum

        }

    }
}

/*
    a    d
    b    e
    c    f

    a / ( a + b + c)    d / (d + e + f)
    b / ( a + b + c)    e / (d + e + f)
    c / ( a + b + c)    f / (d + e + f)

    chain * d a / (a + b + c) / d a = chain(a) * (b + c) / (a + b + c)^2
                                    = chain(a) * [ (b + c) / (a + b + c) ] / (a + b + c)
                                    = chain(a) * [ (a + b + c) - a / (a + b + c) ] / (a + b + c)
                                    = chain(a) * [ (a + b + c) / (a + b + c) - a / (a + b+ c) ] / (a + b + c)
                                    = chain(a) * [ 1 - (a / (a + b + c)) ] / (a + b + c)
                                    = chain(a) * [ 1 - forward(a) ] / sum
    chain * d b / (a + b + c) / d a = chain(b) * -b / (a + b + c)^2
                                    = chain(b) * [ -b / (a + b + c) ] / (a + b + c)
                                    = chain(b) * [ -forward(b) ] / sum
    chain * d c / (a + b + c) / d a = chain(c) * -c / (a + b + c)^2
                                    = chain(c) * [ -c / (a + b + c) ] / (a + b + c)
                                    = chain(c) * [ -forward(c) ] / sum
    = chain(a) * (1 - forward(a)) / sum - chain(b) * forward(b) / sum - chain(c) * forward(c) / sum
    = [ chain(a) * (1 - forward(a)) - chain(b) * forward(b) - chain(c) * forward(c) ] / sum
    = [ chain(a) - chain(a) * forward(a) + chain(b) * -forward(b) + chain(c) * -forward(c) ] / sum

 */

fun backwardNormalization(numberRows: Int, numberColumns: Int, chainEntries: FloatArray, forwardEntries: FloatArray, denominators: FloatArray, result: FloatArray) {
    for (indexColumn in 0 until numberColumns) {
        val startColumn = indexColumn * numberRows

        var productSum = 0.0f

        for (indexRow in 0 until numberRows) {
            val indexEntry = startColumn + indexRow

            val chainEntry = chainEntries[indexEntry]
            val forwardEntry = forwardEntries[indexEntry]

            productSum -= chainEntry * forwardEntry
        }

        for (indexRow in 0 until numberRows) {
            val indexEntry = startColumn + indexRow
            val chainEntry = chainEntries[indexEntry]

            result[indexEntry] = (productSum + chainEntry) / denominators[indexColumn]
        }
    }
}
