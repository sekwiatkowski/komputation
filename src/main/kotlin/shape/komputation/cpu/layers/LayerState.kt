package shape.komputation.cpu.layers

interface ForwardLayerState {

    val forwardResult: FloatArray
    val numberOutputRows: Int
    val numberOutputColumns: Int

}

interface BackwardLayerState {

    val backwardResult: FloatArray
    val numberInputRows: Int
    val numberInputColumns: Int

}

interface LayerState : ForwardLayerState, BackwardLayerState