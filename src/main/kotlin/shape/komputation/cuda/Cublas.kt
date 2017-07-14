package shape.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2.cublasGetVector
import jcuda.jcublas.JCublas2.cublasSetVector
import jcuda.runtime.JCuda.cudaMalloc

fun copyFromHostToDevice(data: DoubleArray, size: Int, pointer : Pointer): Int {

    allocateDeviceMemory(pointer, size)

    return setVector(pointer, data, size)

}

fun allocateDeviceMemory(pointer: Pointer, size : Int) =

    cudaMalloc(pointer, (size * Sizeof.DOUBLE).toLong())

fun setVector(devicePointer: Pointer, data : DoubleArray, size: Int) =

    cublasSetVector(size, Sizeof.DOUBLE, Pointer.to(data), 1, devicePointer, 1)

fun setVectorToZero(devicePointer: Pointer, size: Int) =

    setVector(devicePointer, DoubleArray(size), size)

fun getVector(devicePointer: Pointer, size: Int): DoubleArray {

    val hostVector = DoubleArray(size)

    cublasGetVector(size, Sizeof.DOUBLE, devicePointer, 1, Pointer.to(hostVector), 1)

    return hostVector

}