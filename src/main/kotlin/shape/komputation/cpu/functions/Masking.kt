package shape.komputation.cpu.functions

import java.util.*

fun generateMask(dimension : Int, random : Random, keepProbability: Float) =

    BooleanArray(dimension) {

        random.nextDouble() <= keepProbability

    }