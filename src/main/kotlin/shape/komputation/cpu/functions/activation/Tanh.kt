package shape.komputation.cpu.functions.activation

fun tanh(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        tanh(input[index])

    }

fun tanh(x: Double) =

    (2.0 / (1.0 + Math.exp(-2.0*x))) - 1.0

fun differentiateTanh(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        1 - Math.pow(input[index], 2.0)

    }