package com.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class InputMemory {

    private val deviceData = hashMapOf<Int, Pointer>()
    private val deviceLengths = hashMapOf<Int, Pointer>()
    private val hostMaximumLength = hashMapOf<Int, Int>()

    fun set(id : Int, devicePointer: Pointer, deviceLengths: Pointer, hostMaximumLength : Int) {
        this.deviceData[id] = devicePointer
        this.deviceLengths[id] = deviceLengths
        this.hostMaximumLength[id] = hostMaximumLength
    }

    fun tryToGetData(id : Int) =
        this.deviceData[id]

    fun getData(id: Int) =
        this.deviceData[id]!!

    fun setData(id: Int, deviceData: Pointer) {
        this.deviceData[id] = deviceData
    }

    fun getDeviceLengths(id : Int) =
        this.deviceLengths[id]!!

    fun setDeviceLengths(id : Int, deviceLengths: Pointer) {
        this.deviceLengths[id] = deviceLengths
    }

    fun getHostMaximumLength(id : Int) =
        this.hostMaximumLength[id]!!

    fun setHostMaximumLength(id : Int, maximumLength : Int) {
        this.hostMaximumLength[id] = maximumLength
    }

    fun free() {
        arrayOf(this.deviceData, this.deviceLengths).forEach { map ->
            map.values.forEach { pointer ->
                cudaFree(pointer)
            }
        }

        this.deviceData.clear()
        this.deviceLengths.clear()

        this.hostMaximumLength.clear()
    }

}