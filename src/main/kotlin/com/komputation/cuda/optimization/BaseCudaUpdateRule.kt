package com.komputation.cuda.optimization

import com.komputation.cuda.setIntArray
import com.komputation.instructions.Resourceful
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

abstract class BaseCudaUpdateRule : CudaUpdateRule, Resourceful {

    private val deviceZero = Pointer()
    private val pointerToZero = Pointer.to(this.deviceZero)

    private val deviceCountMap = hashMapOf<Int, Pointer>()

    override fun acquire(maximumBatchSize: Int) {
        setIntArray(intArrayOf(0), 1, this.deviceZero)
    }

    override fun release() {
        cudaFree(this.deviceZero)

        this.deviceCountMap.values.forEach { deviceCount ->
            cudaFree(deviceCount)
        }
    }

    override fun denseUpdate(
        numberParameters: Int,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) {
        val optionalDeviceCount = this.deviceCountMap[numberParameters]

        val deviceCount = if (optionalDeviceCount == null) {

            val deviceCount = Pointer()
            setIntArray(intArrayOf(numberParameters), 1, deviceCount)

            this.deviceCountMap[numberParameters] = deviceCount

            deviceCount
        }
        else {
            optionalDeviceCount
        }

        this.launchKernel(
            1,
            this.pointerToZero,
            Pointer.to(deviceCount),
            pointerToParameters,
            pointerToGradient)
    }

    override fun sparseUpdate(
        numberParameters: Int,
        pointerToParameterIndices: Pointer,
        pointerToCounts: Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) {
        this.launchKernel(
            numberParameters,
            pointerToParameterIndices,
            pointerToCounts,
            pointerToParameters,
            pointerToGradient)
    }

    abstract fun launchKernel(
        numberParameters: Int,
        pointerToParameterIndices: Pointer,
        pointerToCounts : Pointer,
        pointerToParameters: Pointer,
        pointerToGradient: Pointer) : Int

}