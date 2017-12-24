package com.komputation.cuda.layers.entry

import jcuda.Pointer
import com.komputation.cuda.setIntArray
import com.komputation.instructions.Resourceful
import jcuda.runtime.JCuda.cudaFree
import com.komputation.cuda.computeDeviceIntArraySize
import com.komputation.cuda.kernels.Kernel
import com.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration

class CudaHashing(
    private val maximumColumns: Int,
    private val sizeMultiplier : Int,
    private val createHashingKernel: () -> Kernel,
    private val createFillTwoIntegerArraysKernel: () -> Kernel,
    private val numberMultiprocessors : Int,
    private val numberResidentWarps: Int,
    private val warpSize: Int,
    private val maximumNumberThreadsPerBlock : Int) : Resourceful {

    private val deviceMapping = Pointer()
    fun getDeviceMapping() = this.deviceMapping
    private val pointerToMapping = Pointer.to(this.deviceMapping)
    fun getPointerToMapping() = this.pointerToMapping

    private val deviceHashTable = Pointer()
    fun getDeviceHashTable() = this.deviceHashTable
    private val pointerToHashTable = Pointer.to(this.deviceHashTable)
    fun getPointerToHashTable() = this.pointerToHashTable

    private val deviceCounts = Pointer()
    fun getDeviceCounts() = this.deviceCounts
    private val pointerToCounts = Pointer.to(this.deviceCounts)
    fun getPointerToCounts() = this.pointerToCounts

    private var hashingKernel : Kernel? = null
    private var fillTwoIntegerArraysKernel : Kernel? = null
    private var maximumBatchSize = -1

    private val pointerToMinusOne = Pointer.to(intArrayOf(-1))
    private val pointerToZero = Pointer.to(intArrayOf(0))

    private val hashTableSize = intArrayOf(-1)
    fun getHashTableSize() = this.hashTableSize[0]
    private val pointerToHashTableSize = Pointer.to(this.hashTableSize)

    private val reset_numberIterations = intArrayOf(-1)
    private val reset_pointerToNumberIterations = Pointer.to(this.reset_numberIterations)
    private var reset_numberBlocks = -1
    private var reset_numberThreadsPerBlock = -1

    override fun acquire(maximumBatchSize: Int) {
        this.maximumBatchSize = maximumBatchSize

        val hashTableSize = this.maximumBatchSize * this.sizeMultiplier * this.maximumColumns
        this.hashTableSize[0] = hashTableSize

        setIntArray(IntArray(hashTableSize) { -1 }, hashTableSize, this.deviceHashTable)
        setIntArray(IntArray(hashTableSize) { 0 }, hashTableSize, this.deviceCounts)

        val mappingSize = this.maximumBatchSize * this.maximumColumns
        setIntArray(IntArray(mappingSize) { -1 }, mappingSize, this.deviceMapping)

        this.fillTwoIntegerArraysKernel = this.createFillTwoIntegerArraysKernel()
        this.hashingKernel = this.createHashingKernel()

        val resetConfiguration = computeEntrywiseLaunchConfiguration(this.hashTableSize[0], this.numberMultiprocessors, this.numberResidentWarps, this.warpSize, this.maximumNumberThreadsPerBlock)
        this.reset_numberIterations[0] = resetConfiguration.numberIterations
        this.reset_numberBlocks = resetConfiguration.numberBlocks
        this.reset_numberThreadsPerBlock = resetConfiguration.numberThreadsPerBlock
    }

    fun reset() {
        this.fillTwoIntegerArraysKernel!!.launch(
            Pointer.to(
                this.pointerToHashTableSize,
                this.reset_pointerToNumberIterations,
                this.pointerToHashTable,
                this.pointerToMinusOne,
                this.pointerToCounts,
                this.pointerToZero
            ),
            this.reset_numberBlocks,
            1,
            this.reset_numberThreadsPerBlock,
            0)
    }

    fun hash(pointerToIndices : Pointer) {
        this.hashingKernel!!.launch(
            Pointer.to(
                pointerToIndices,
                this.pointerToHashTableSize,
                this.pointerToHashTable,
                this.pointerToCounts,
                this.pointerToMapping
            ),
            this.maximumBatchSize,
            1,
            this.maximumColumns,
            computeDeviceIntArraySize(this.hashTableSize[0]).toInt())
    }

    override fun release() {
        this.maximumBatchSize = -1

        cudaFree(this.deviceMapping)
        cudaFree(this.deviceHashTable)
        cudaFree(this.deviceCounts)

        this.hashingKernel!!.destroy()
        this.fillTwoIntegerArraysKernel!!.destroy()

        this.reset_numberIterations[0] = -1
        this.reset_numberBlocks = -1
        this.reset_numberThreadsPerBlock = -1
    }

}