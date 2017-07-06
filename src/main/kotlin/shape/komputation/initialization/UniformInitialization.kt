package shape.komputation.initialization

import java.util.*

class UniformInitialization internal constructor(
    private val random: Random,
    private val min: Double = 0.0,
    max: Double = 1.0) : InitializationStrategy {

    private val difference = max - min

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        random.nextDouble() * difference - Math.abs(min)

}

fun uniformInitialization(random: Random, min: Double = 0.0, max: Double = 1.0) =

    UniformInitialization(random, min, max)