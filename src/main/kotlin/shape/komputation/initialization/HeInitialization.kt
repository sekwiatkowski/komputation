package shape.komputation.initialization

import java.util.*

class HeInitialization internal constructor(private val random: Random) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        random.nextGaussian() * Math.sqrt(1.0.div(numberIncoming.toDouble()))

}

fun heInitialization(random: Random) =

    HeInitialization(random)