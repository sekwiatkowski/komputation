package shape.komputation.cpu.functions

import java.util.*

fun generateMask(dimension : Int, random : Random, keepProbability: Double) =

    BooleanArray(dimension) {

        random.nextDouble() <= keepProbability

    }