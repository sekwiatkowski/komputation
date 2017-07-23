package shape.komputation.cpu.functions.activation

import shape.komputation.matrix.FloatMath

fun sigmoid(input: FloatArray, result : FloatArray, numberEntries : Int) {

    for (index in 0..numberEntries - 1) {

        result[index] = sigmoid(input[index])

    }

}

fun sigmoid(x: Float) =

    1.0f / (1.0f + FloatMath.exp(-x))

fun differentiateSigmoid(forwardEntries: FloatArray) =

    FloatArray(forwardEntries.size) { index ->

        val forward = forwardEntries[index]

        forward * (1.0f - forward)

    }