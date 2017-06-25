package shape.komputation.functions.activation

fun tanh(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        tanh(input[index])

    }

fun tanh(x: Double) =

    (2.0 / (1.0 + Math.exp(-2.0*x))) - 1.0

fun differentiateTanh(forwardEntries: DoubleArray) =

    DoubleArray(forwardEntries.size) { index ->

        1 - Math.pow(forwardEntries[index], 2.0)

    }