package shape.komputation.optimization

fun updateSparsely(vectors: Array<DoubleArray>, numberVectors : Int, dimension : Int, size : Int, ids: IntArray, counts : DoubleArray, gradients: Array<DoubleArray>, rule : UpdateRule) {

    for (index in 0..size - 1) {

        val id = ids[index]
        val count = counts[index]
        val scalingFactor = 1.0.div(count)
        val vector = vectors[id]
        val gradient = gradients[index]

        for (indexDimension in 0..dimension - 1) {

            val indexEntry = id + indexDimension * numberVectors
            val current = vector[indexDimension]
            val derivative = scalingFactor * gradient[indexDimension]

            val updated = rule.apply(indexEntry, current, derivative)

            vector[indexDimension] = updated

        }

    }

}