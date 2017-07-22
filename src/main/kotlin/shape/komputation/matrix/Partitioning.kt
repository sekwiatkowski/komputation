package shape.komputation.matrix

fun partitionIndices(size : Int, batchSize : Int): Array<Array<Int>> {

    var count = 0

    val numberBatches = IntMath.ceil(size.toDouble().div(batchSize.toDouble()))

    val batches = Array(numberBatches) {

        val currentBatchSize = Math.min(size - count, batchSize)

        Array(currentBatchSize) {

            count++

        }

    }

    return batches

}