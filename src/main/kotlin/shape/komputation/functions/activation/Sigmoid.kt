package shape.komputation.functions.activation

fun sigmoid(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        sigmoid(input[index])

    }

fun sigmoid(x: Double) =

    1.0 / (1.0 + Math.exp(-x))

fun differentiateSigmoid(forwardEntries: DoubleArray) =

    DoubleArray(forwardEntries.size) { index ->

        val forward = forwardEntries[index]

        forward * (1 - forward)

    }