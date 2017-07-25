package shape.komputation.cpu.functions.activation

import shape.komputation.cpu.functions.hadamard
import shape.komputation.matrix.FloatMath

fun exponentiate(input: FloatArray, result : FloatArray, numberEntries : Int) {

    for(index in 0..numberEntries-1) {

        result[index] = FloatMath.exp(input[index])

    }

}

fun backwardExponentiation(forwardEntries: FloatArray, chain : FloatArray, result : FloatArray, numberEntries : Int) {

    hadamard(forwardEntries, chain, result, numberEntries)

}