package shape.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.*
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice

fun setVector(data: DoubleArray, size: Int, pointer : Pointer): Int {

    allocateDeviceMemory(pointer, size)

    return copyFromHostToDevice(pointer, data, size)

}

fun allocateDeviceMemory(pointer: Pointer, size : Int) =

    cudaMalloc(pointer, (size * Sizeof.DOUBLE).toLong())

fun copyFromHostToDevice(devicePointer: Pointer, data : DoubleArray, size: Int) =

    cudaMemcpy(devicePointer, Pointer.to(data), (size * Sizeof.DOUBLE).toLong(), cudaMemcpyHostToDevice)

fun setVectorToZero(devicePointer: Pointer, size: Int) =

    cudaMemset(devicePointer, 0, (Sizeof.DOUBLE * size).toLong())

fun getVector(devicePointer: Pointer, size: Int): DoubleArray {

    val hostVector = DoubleArray(size)

    cudaMemcpy(Pointer.to(hostVector), devicePointer, (size * Sizeof.DOUBLE).toLong(), cudaMemcpyDeviceToHost)

    return hostVector

}