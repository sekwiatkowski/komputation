package shape.komputation.matrix

fun partitionIndices(size : Int, batchSize : Int): Array<Array<Int>> {

    var count = 0

    val numberBatches = Math.ceil(size.toDouble().div(batchSize.toDouble())).toInt()

    val batches = Array(numberBatches) { index ->

        val currentBatchSize = Math.min(size - count, batchSize)

        Array(currentBatchSize) {

            count++

        }

    }

    return batches

}