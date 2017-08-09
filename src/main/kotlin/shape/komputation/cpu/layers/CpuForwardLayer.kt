package shape.komputation.cpu.layers

interface CpuForwardLayer : LayerState {

    fun forward(withinBatch : Int, numberInputColumns : Int, input: FloatArray, isTraining : Boolean) : FloatArray

    fun backward(withinBatch : Int, chain : FloatArray) : FloatArray

}