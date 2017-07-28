package shape.komputation.layers

interface Resourceful {

    fun acquire(maximumBatchSize : Int)

    fun release()

}