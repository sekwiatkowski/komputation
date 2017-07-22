package shape.komputation.cpu.functions

import java.util.*

fun extractStep(chainEntries : FloatArray, indexStep : Int, stepSize : Int): FloatArray {

    val start = indexStep * stepSize
    val end = start + stepSize

    return Arrays.copyOfRange(chainEntries, start, end)

}