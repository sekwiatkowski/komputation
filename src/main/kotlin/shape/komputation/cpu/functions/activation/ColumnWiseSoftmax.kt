package shape.komputation.cpu.functions.activation

import shape.komputation.matrix.FloatMath
import java.util.*

fun columnWiseSoftmax(input: FloatArray, numberRows : Int, numberColumns : Int, result : FloatArray) {

    for (indexColumn in 0..numberColumns - 1) {

        val start = indexColumn * numberRows

        var sum = 0.0f
        for (indexRow in 0..numberRows - 1) {

            val indexEntry = start + indexRow

            val exponentiation = FloatMath.exp(input[indexEntry])
            sum += exponentiation

            result[indexEntry] = exponentiation

        }

        for (indexRow in 0..numberRows - 1) {

            result[start + indexRow] /= sum

        }

    }

}

fun backwardColumnWiseSoftmax(numberForwardRows: Int, numberForwardColumns: Int, forwardEntries: FloatArray, chainEntries: FloatArray, result : FloatArray) {

    for (indexColumn in 0..numberForwardColumns - 1) {

        val start = indexColumn * numberForwardRows
        val end = start + numberForwardRows

        val forwardColumn = Arrays.copyOfRange(forwardEntries, start, end)
        val chainColumn = Arrays.copyOfRange(chainEntries, start, end)

        for (outerIndexRow in 0..numberForwardRows - 1) {

            var derivative = 0.0f

            val prediction = forwardColumn[outerIndexRow]

            for (innerIndexRow in 0..numberForwardRows - 1) {

                val chainEntry = chainColumn[innerIndexRow]

                // i == j
                if (outerIndexRow == innerIndexRow) {

                    derivative += chainEntry * prediction * (1 - prediction)

                }
                // i != j
                else {

                    val otherPrediction = forwardColumn[innerIndexRow]

                    derivative += chainEntry * (-prediction * otherPrediction)

                }

            }

            result[start + outerIndexRow] = derivative

        }

    }

}
