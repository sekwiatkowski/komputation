package com.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree


data class DuplicateMemory(
    val deviceFirstOccurrences : Pointer,
    val deviceOtherOccurrences : Pointer,
    val deviceOccurrencePositions : Pointer)

class InputMemory {

    private val deviceData = hashMapOf<Int, Pointer>()
    private val deviceLengths = hashMapOf<Int, Pointer>()
    private val deviceCounts = hashMapOf<Int, Pointer>()

    private val numberDuplicates = hashMapOf<Int, Int>()
    private val deviceDuplicateMemory = hashMapOf<Int, DuplicateMemory>()

    fun setFixedLengthData(id : Int, devicePointer: Pointer) {
        this.deviceData[id] = devicePointer
    }

    fun setVariableLengthData(id : Int, devicePointer: Pointer, deviceLengths: Pointer) {
        this.deviceData[id] = devicePointer
        this.deviceLengths[id] = deviceLengths
    }

    fun setWithDuplicates(id: Int, numberDuplicates : Int, deviceDuplicateMemory: DuplicateMemory) {
        this.numberDuplicates[id] = numberDuplicates
        this.deviceDuplicateMemory[id] = deviceDuplicateMemory
    }

    fun setWithoutDuplicates(id : Int) {
        this.numberDuplicates[id] = 0
    }

    fun setCounts(id : Int, deviceCounts : Pointer) {
        this.deviceCounts[id] = deviceCounts
    }

    fun tryToGetData(id : Int) =
        this.deviceData[id]

    fun getData(id: Int) =
        this.deviceData[id]!!

    fun getDeviceLengths(id : Int) =
        this.deviceLengths[id]!!

    fun getDeviceCounts(id : Int) =
        this.deviceCounts[id]!!

    fun getNumberDuplicates(id: Int) =
        this.numberDuplicates[id]!!

    fun getDuplicateMemory(id : Int) =
        this.deviceDuplicateMemory[id]!!

    fun free() {
        arrayOf(this.deviceData, this.deviceLengths, this.deviceCounts).forEach { map ->
            map.values.forEach { pointer ->
                cudaFree(pointer)
            }
        }

        this.deviceDuplicateMemory.values.forEach { (firstOccurrences, otherOccurrences, occurrencePositions) ->
            cudaFree(firstOccurrences)
            cudaFree(otherOccurrences)
            cudaFree(occurrencePositions)
        }

        this.deviceData.clear()
        this.deviceLengths.clear()
        this.deviceCounts.clear()

        this.deviceDuplicateMemory.clear()
        this.numberDuplicates.clear()
    }

}