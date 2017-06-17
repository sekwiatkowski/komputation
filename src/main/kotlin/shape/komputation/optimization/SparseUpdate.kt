package shape.komputation.optimization

import shape.komputation.matrix.DoubleMatrix


/*
    ids: 0, 1, 0

    gradients: emb(0)_0, emb(0)_1, emb(1)_0, emb(1)_1, emb(2)_0, emb(2)_1
 */

fun updateSparsely(data: Array<DoubleArray>, dimension : Int, idBatch: Array<IntArray>, gradientBatch: Array<DoubleArray>, rule : UpdateRule) {

    val batchSize = idBatch.size
    val scalingFactor = 1.0.div(batchSize)

    for (index in 0..batchSize - 1) {

        val ids = idBatch[index]
        val gradients = gradientBatch[index]

        val numberInstances = ids.size

        for (indexInstance in 0..numberInstances -1) {

            val id = ids[indexInstance]
            val instance = data[id]

            for (indexDimension in 0..dimension - 1) {

                val current = instance[indexDimension]

                val derivative = gradients[indexInstance + indexDimension * numberInstances]

                val updated = rule.apply(id + indexDimension * data.size, current, scalingFactor * derivative)

                instance[indexDimension] = updated

            }

        }

    }

}