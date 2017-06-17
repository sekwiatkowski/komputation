package shape.komputation.functions.activation

import shape.komputation.matrix.DoubleMatrix
import java.util.*

fun softmax(input: DoubleArray, numberRows : Int, numberColumns : Int): DoubleMatrix {

    var counter = 0

    val results = DoubleArray(numberRows)

    for (indexColumn in 0..numberColumns - 1) {

        var sum = 0.0

        for (indexRow in 0..numberRows - 1) {

            val exponentiation = Math.exp(input[counter + indexRow])
            sum += exponentiation

            results[indexRow] = exponentiation

        }

        for (indexRow in 0..numberRows - 1) {

            results[indexRow] = results[indexRow].div(sum)

        }

        counter += numberRows

    }

    return DoubleMatrix(numberRows, numberColumns, results)

}

fun backwardSoftmax(numberForwardRows: Int, numberForwardColumns: Int, forwardEntries: DoubleArray, chainEntries: DoubleArray): DoubleArray {

    val gradient = DoubleArray(numberForwardRows * numberForwardColumns)

    for (indexColumn in 0..numberForwardColumns - 1) {

        val start = indexColumn * numberForwardRows
        val end = start + numberForwardRows

        val forwardColumn = Arrays.copyOfRange(forwardEntries, start, end)
        val chainColumn = Arrays.copyOfRange(chainEntries, start, end)

        for (outerIndexRow in 0..numberForwardRows - 1) {

            var derivative = 0.0

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

            gradient[start + outerIndexRow] = derivative

        }

    }

    return gradient
}
