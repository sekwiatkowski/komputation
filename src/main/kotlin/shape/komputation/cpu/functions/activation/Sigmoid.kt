package shape.komputation.cpu.functions.activation

import shape.komputation.matrix.FloatMath

fun sigmoid(input: FloatArray) =

    FloatArray(input.size) { index ->

        sigmoid(input[index])

    }

fun sigmoid(x: Float) =

    1.0f / (1.0f + FloatMath.exp(-x))

fun differentiateSigmoid(forwardEntries: FloatArray) =

    FloatArray(forwardEntries.size) { index ->

        val forward = forwardEntries[index]

        forward * (1.0f - forward)

    }