package shape.komputation.functions.activation

fun tanh(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        tanh(input[index])

    }

fun tanh(x: Double) =

    (2.0 / (1.0 + Math.exp(-2.0*x))) - 1.0

fun backwardTanh(forwardEntries: DoubleArray, chainEntries: DoubleArray) =

    DoubleArray(forwardEntries.size) { index ->

        val forward = forwardEntries[index]
        val chainEntry = chainEntries[index]

        val dActivationWrtPreActivation = 1 - Math.pow(forward, 2.0)

        val derivative = chainEntry * dActivationWrtPreActivation

        derivative

    }