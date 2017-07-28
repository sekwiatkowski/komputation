package shape.komputation.matrix

fun partitionIndices(size : Int, batchSize : Int): Array<IntArray> {

    val numberBatches = IntMath.ceil(size.toDouble().div(batchSize.toDouble()))

    return Array(numberBatches) { indexBatch ->

        val start = indexBatch * batchSize

        IntArray(batchSize) { index -> start + index }

    }

}