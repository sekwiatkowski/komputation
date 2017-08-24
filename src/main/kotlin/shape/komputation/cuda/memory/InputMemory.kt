package shape.komputation.cuda.memory

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class InputMemory {

    private val deviceData = hashMapOf<Int, Pointer>()
    private val totalNumbersOfColumns = hashMapOf<Int, Int>()

    fun tryToGetData(id: Int) =

        this.deviceData[id]

    fun getDeviceData(id : Int) =

        this.deviceData[id]!!

    fun getTotalNumbersOfColumns(id: Int) =

        this.totalNumbersOfColumns[id]!!

    fun setData(id : Int, pointer: Pointer) {

        this.deviceData[id] = pointer

    }


    fun setTotalNumberOfColumns(id : Int, number : Int) {

        this.totalNumbersOfColumns[id] = number

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