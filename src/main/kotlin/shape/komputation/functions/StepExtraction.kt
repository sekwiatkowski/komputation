package shape.komputation.functions

import java.util.*

fun extractStep(chainEntries : DoubleArray, indexStep : Int, stepSize : Int): DoubleArray {

    val start = indexStep * stepSize
    val end = start + stepSize

    return Arrays.copyOfRange(chainEntries, start, end)

}