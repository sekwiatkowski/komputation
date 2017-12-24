package com.komputation.cpu.optimization

import com.komputation.cpu.functions.scale

fun updateSparsely(vectors: Array<FloatArray>, dimension : Int, numberUpdates: Int, parameterIndices: IntArray, counts : FloatArray, gradients: Array<FloatArray>, rule : UpdateRule) {

    for (indexUpdate in 0 until numberUpdates) {

        val parameterIndex = parameterIndices[indexUpdate]
        val parameter = vectors[parameterIndex]
        val firstParameterEntryIndex = parameterIndex * dimension

        val gradient = gradients[indexUpdate]
        val count = counts[indexUpdate]
        val scalingFactor = 1f.div(count)
        scale(gradient, scalingFactor, gradient, dimension)

        rule.updateSparsely(firstParameterEntryIndex, parameter, gradient, dimension)

    }

}