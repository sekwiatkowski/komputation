package shape.komputation.initialization

import java.util.*

fun uniformInitialization(random: Random, min: Double = 0.0, max: Double = 1.0): InitializationStrategy {

    val difference = max - min

    return { _, _ -> random.nextDouble() * difference - Math.abs(min) }

}