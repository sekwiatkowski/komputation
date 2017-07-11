package shape.komputation.matrix

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.cublasSetVector
import jcuda.runtime.JCuda.cudaMalloc

fun copyFromHostToDevice(data: DoubleArray, dimension : Int): Pointer {

    val pointer = Pointer()
    cudaMalloc(pointer, (dimension * Sizeof.DOUBLE).toLong())
    cublasSetVector(dimension, Sizeof.DOUBLE, Pointer.to(data), 1, pointer, 1)

    return pointer

}