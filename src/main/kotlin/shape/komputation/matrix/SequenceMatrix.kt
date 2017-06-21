package shape.komputation.matrix

val EMPTY_SEQUENCE_MATRIX = SequenceMatrix(0, 0, 0, doubleArrayOf())

fun zeroSequenceMatrix(numberSteps: Int, stepRows: Int, stepColumns: Int = 1) =

    SequenceMatrix(numberSteps, stepRows, stepColumns, DoubleArray(numberSteps * stepRows * stepColumns))

fun sequence(stepRows : Int, vararg steps : DoubleArray): SequenceMatrix {

    return sequence(stepRows, 1, *steps)

}

fun step(vararg entries: Double) =

    doubleArrayOf(*entries)

fun sequence(stepRows : Int, numberStepColumns: Int, vararg steps: DoubleArray): SequenceMatrix {

    val numberSteps = steps.size

    val entries = DoubleArray(stepRows * numberSteps * numberStepColumns)

    var count = 0

    for (step in steps) {

        for (entry in step) {

            entries[count++] = entry

        }

    }

    return SequenceMatrix(numberSteps, stepRows, numberStepColumns, entries)

}

fun sequence(numberSteps : Int, stepRows : Int, generate : (Int) -> DoubleArray): SequenceMatrix {

    val steps = Array<DoubleArray>(numberSteps, generate)

    return sequence(stepRows, 1, *steps)

}