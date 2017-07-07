package shape.komputation.optimization

fun updateSparsely(vectors: Array<DoubleArray>, dimension : Int, size : Int, ids: IntArray, counts : DoubleArray, gradients: Array<DoubleArray>, rule : UpdateRule) {

    for (index in 0..size - 1) {

        val id = ids[index]
        val count = counts[index]
        val vector = vectors[id]
        val gradient = gradients[index]

        val scalingFactor = 1.0.div(count)

        for (indexDimension in 0..dimension - 1) {

            val indexEntry = indexDimension + id * dimension
            val current = vector[indexDimension]
            val derivative = scalingFactor * gradient[indexDimension]

            val updated = rule.apply(indexEntry, current, derivative)

            vector[indexDimension] = updated

        }

    }

}