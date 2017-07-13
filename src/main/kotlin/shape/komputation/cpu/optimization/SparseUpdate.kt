package shape.komputation.cpu.optimization

import shape.komputation.functions.scale

fun updateSparsely(vectors: Array<DoubleArray>, dimension : Int, size : Int, ids: IntArray, counts : DoubleArray, gradients: Array<DoubleArray>, rule : UpdateRule) {

    for (index in 0..size - 1) {

        val id = ids[index]
        val start = id * dimension

        val parameters = vectors[id]

        val gradient = gradients[index]
        val count = counts[index]
        val scalingFactor = 1.0.div(count)
        val scaledGradient = scale(gradient, scalingFactor)

        rule.updateSparsely(start, parameters, scaledGradient, scaledGradient.size)

    }

}