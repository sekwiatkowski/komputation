package shape.konvolution

import java.util.*

fun createUniformInitializer(random: Random, min: Double = 0.0, max: Double = 1.0): () -> Double {

    val difference = max - min

    return { random.nextDouble() * difference - Math.abs(min) }

}

fun createGaussianInitializer(random: Random): () -> Double {

    return { random.nextGaussian() }

}

fun initializeMatrix(generateEntry : () -> Double, numberRows : Int, numberColumns: Int) =

    createRealMatrix(numberRows, numberColumns).let { matrix ->

        for (indexRow in 0..numberRows - 1) {

            for (indexColumns in 0..numberColumns - 1) {

                matrix.set(indexRow, indexColumns, generateEntry())

            }

        }

        matrix

    }

fun initializeBias(generateEntry : () -> Double, numberRows : Int) =

    initializeMatrix(generateEntry, numberRows, 1)