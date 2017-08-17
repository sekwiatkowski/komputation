package shape.komputation.cuda

import jcuda.Pointer

interface CudaForwardState {

    val deviceForwardResult: Pointer
    val numberOutputRows: Int
    val numberOutputColumns: Int

}

interface CudaBackwardState {

    val deviceBackwardResult: Pointer
    val numberInputRows: Int
    val numberInputColumns: Int

}

interface CudaLayerState : CudaForwardState, CudaBackwardState