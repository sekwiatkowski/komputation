package shape.komputation.cuda

import jcuda.Pointer

interface CudaForwardState {

    val deviceForwardResult: Pointer
    val deviceNumberOutputColumns : Pointer
    val numberOutputRows : Int
    val maximumOutputColumns : Int

}

interface CudaBackwardState {

    val deviceBackwardResult: Pointer
    val deviceNumberInputColumns : Pointer
    val numberInputRows : Int
    val maximumInputColumns : Int

}

interface CudaLayerState : CudaForwardState, CudaBackwardState