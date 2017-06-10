package shape.konvolution.optimization

import shape.konvolution.matrix.RealMatrix

fun optimizeSparsely(rows: Array<DoubleArray>, rowIndices: IntArray, updates: RealMatrix) {

    for ((updateIndex, rowIndex) in rowIndices.withIndex()) {

        val embedding = rows[rowIndex]

        for (indexColumn in 0..embedding.size - 1) {

            embedding[indexColumn] = updates.get(updateIndex, indexColumn)

        }

    }

}