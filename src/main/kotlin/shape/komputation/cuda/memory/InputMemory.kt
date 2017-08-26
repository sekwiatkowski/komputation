package shape.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class InputMemory {

    private val deviceData = hashMapOf<Int, Pointer>()
    private val totalNumbersOfColumns = hashMapOf<Int, Int>()
    private val deviceLengths = hashMapOf<Int, Pointer>()

    fun tryToGetData(id: Int) =

        this.deviceData[id]

    fun getDeviceData(id : Int) =

        this.deviceData[id]!!

    fun setData(id : Int, pointer: Pointer) {

        this.deviceData[id] = pointer

    }

    fun getTotalNumbersOfColumns(id: Int) =

        this.totalNumbersOfColumns[id]!!

    fun setTotalNumberOfColumns(id : Int, number : Int) {

        this.totalNumbersOfColumns[id] = number

    }

    fun getLengths(id : Int) =

        this.deviceLengths[id]!!

    fun setLengths(id : Int, pointer : Pointer) {

        this.deviceLengths[id] = pointer

    }

    fun free() {

        arrayOf(this.deviceData).forEach { map ->

            map.values.forEach { pointer ->

                cudaFree(pointer)

            }

        }

        this.deviceData.clear()
        this.totalNumbersOfColumns.clear()

    }

}