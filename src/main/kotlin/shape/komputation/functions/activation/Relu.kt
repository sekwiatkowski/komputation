package shape.komputation.functions.activation

fun relu(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        relu(input[index])

    }

fun relu(entry: Double) =

    Math.max(entry, 0.0)
