package shape.komputation.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda.*
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
import jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice

fun computeDeviceByteSize(arraySize : Int) =

    (arraySize * Sizeof.FLOAT).toLong()

fun setVector(data: FloatArray, size: Int, pointer : Pointer): Int {

    allocateDeviceMemory(pointer, size)

    return copyFromHostToDevice(pointer, data, size)

}

fun allocateDeviceMemory(pointer: Pointer, size : Int) =

    cudaMalloc(pointer, computeDeviceByteSize(size))

fun copyFromHostToDevice(devicePointer: Pointer, data : FloatArray, size: Int) =

    cudaMemcpy(devicePointer, Pointer.to(data), computeDeviceByteSize(size), cudaMemcpyHostToDevice)

fun setVectorToZero(devicePointer: Pointer, size: Int) =

    cudaMemset(devicePointer, 0, computeDeviceByteSize(size))

fun getVector(devicePointer: Pointer, size: Int): FloatArray {

    val hostVector = FloatArray(size)

    cudaMemcpy(Pointer.to(hostVector), devicePointer, computeDeviceByteSize(size), cudaMemcpyDeviceToHost)

    return hostVector

}