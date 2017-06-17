package shape.komputation.functions.activation

fun sigmoid(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        sigmoid(input[index])

    }

fun sigmoid(x: Double) =

    1.0 / (1.0 + Math.exp(-x))

fun backwardSigmoid(forwardEntries: DoubleArray, chainEntries: DoubleArray) =

    DoubleArray(forwardEntries.size) { index ->

        val forward = forwardEntries[index]
        val chainEntry = chainEntries[index]

        val dActivationWrtPreActivation = forward * (1 - forward)

        val derivative = chainEntry * dActivationWrtPreActivation

        derivative

    }