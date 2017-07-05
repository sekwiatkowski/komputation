package shape.komputation.initialization

import java.util.*

fun createHeInitialization(random: Random, inputDimension : Int): InitializationStrategy {

    val variance = 1.0.div(inputDimension.toDouble())
    val standardDeviation = Math.sqrt(variance)

    return { _, _ -> random.nextGaussian() * standardDeviation }

}