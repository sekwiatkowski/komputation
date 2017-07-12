package shape.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.JCublas2
import jcuda.jcublas.JCublas2.cublasSetVector
import jcuda.runtime.JCuda.cudaMalloc

fun copyFromHostToDevice(data: DoubleArray, size: Int): Pointer {

    val pointer = Pointer()

    allocateDeviceMemory(pointer, size)

    setVector(pointer, data, size)

    return pointer

}

fun allocateDeviceMemory(pointer: Pointer, size : Int) {

    cudaMalloc(pointer, (size * Sizeof.DOUBLE).toLong())

}

fun setVector(devicePointer: Pointer, data : DoubleArray, size: Int) {

    cublasSetVector(size, Sizeof.DOUBLE, Pointer.to(data), 1, devicePointer, 1)

}

fun setVectorToZero(devicePointer: Pointer, size: Int) {

    setVector(devicePointer, DoubleArray(size), size)

}

fun getVector(devicePointer: Pointer, size: Int): DoubleArray {

    val hostVector = DoubleArray(size)

    JCublas2.cublasGetVector(size, Sizeof.DOUBLE, devicePointer, 1, Pointer.to(hostVector), 1)

    return hostVector

}