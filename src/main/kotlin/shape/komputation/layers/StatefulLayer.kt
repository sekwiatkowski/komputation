package shape.komputation.layers

interface StatefulLayer {

    fun startForward()

    fun finishBackward()

}