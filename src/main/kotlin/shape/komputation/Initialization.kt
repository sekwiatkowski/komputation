package shape.komputation

import shape.komputation.matrix.createRealMatrix
import java.util.*

fun createUniformInitializer(random: Random, min: Double = 0.0, max: Double = 1.0): () -> Double {

    val difference = max - min

    return { random.nextDouble() * difference - Math.abs(min) }

}

fun createGaussianInitializer(random: Random): () -> Double {

    return { random.nextGaussian() }

}

fun createConstantInitializer(constant: Double): () -> Double {

    return { constant }

}

fun initializeRow(generateEntry: () -> Double, numberColumns: Int) =

    DoubleArray(numberColumns) { generateEntry() }

fun initializeMatrix(generateEntry : () -> Double, numberRows : Int, numberColumns: Int) =

    Array(numberRows) {

            initializeRow(generateEntry, numberColumns)

        }
        .let { rows ->

            createRealMatrix(*rows)
        }


fun initializeRowVector(generateEntry : () -> Double, dimension: Int) =

    initializeMatrix(generateEntry, dimension, 1)