package shape.komputation.initialization

import java.util.*

fun createGaussianInitializer(random: Random, mean : Double = 0.0, standardDeviation : Double = 1.0): InitializationStrategy {

    return { _, _ -> random.nextGaussian() * standardDeviation + mean }

}
