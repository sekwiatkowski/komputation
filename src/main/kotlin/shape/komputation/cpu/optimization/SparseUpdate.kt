package shape.komputation.cpu.optimization

import shape.komputation.cpu.functions.scale

fun updateSparsely(vectors: Array<FloatArray>, dimension : Int, size : Int, ids: IntArray, counts : FloatArray, gradients: Array<FloatArray>, rule : UpdateRule) {

    for (index in 0..size - 1) {

        val id = ids[index]
        val start = id * dimension

        val parameters = vectors[id]

        val gradient = gradients[index]
        val count = counts[index]
        val scalingFactor = 1.0f.div(count)
        scale(gradient, scalingFactor, dimension)

        rule.updateSparsely(start, parameters, gradient, dimension)

    }

}