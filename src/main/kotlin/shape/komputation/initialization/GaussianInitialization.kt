package shape.komputation.initialization

import java.util.*

class GaussianInitialization internal constructor(
    private val random: Random, private val mean : Double = 0.0, private val standardDeviation : Double = 1.0) : InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        random.nextGaussian() * standardDeviation + mean


}

fun gaussianInitialization(random: Random, mean : Double = 0.0, standardDeviation : Double = 1.0) =

    GaussianInitialization(random, mean, standardDeviation)