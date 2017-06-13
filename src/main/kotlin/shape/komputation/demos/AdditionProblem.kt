package shape.komputation.demos

import shape.komputation.matrix.RealMatrix
import shape.komputation.matrix.createRealMatrix
import java.util.*

fun main(args: Array<String>) {

    val random = Random(1)
    val length = 100
    val numberExamples = 100_000

    val inputs = Array(numberExamples) { generateInput(random, 100) }

    val targets = Array(numberExamples) { indexInput ->

        calculateSolution(inputs[indexInput])

    }


}

private fun generateInput(random: Random, length: Int): RealMatrix {

    val example = createRealMatrix(2, length)

    for (indexColumn in 0..length - 1) {
        example.set(0, indexColumn, random.nextDouble())
    }

    val firstIndex = random.nextInt(length)
    val secondIndex = random.nextInt(length).let { candidate ->

        if (candidate == firstIndex) {

            if (firstIndex == length - 1) {
                firstIndex - 1
            }
            else {
                firstIndex + 1
            }

        }
        else {

            candidate
        }

    }

    example.set(1, firstIndex, 1.0)
    example.set(1, secondIndex, 1.0)

    return example

}

private fun calculateSolution(input: RealMatrix): Double {

    val solution = (0..input.numberColumns() - 1)
        .sumByDouble { input.get(0, it) * input.get(1, it) }

    return solution
}
