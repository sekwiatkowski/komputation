package shape.komputation.cpu.functions.activation

import shape.komputation.matrix.FloatMath

fun tanh(input: FloatArray) =

    FloatArray(input.size) { index ->

        FloatMath.tanh(input[index])

    }

fun differentiateTanh(input: FloatArray) =

    FloatArray(input.size) { index ->

        1.0f - (input[index] * input[index])

    }