package com.komputation.cuda.layers.entry

import com.komputation.cuda.getIntArray
import com.komputation.cuda.kernels.ArrayKernels
import com.komputation.cuda.kernels.HashtableKernels
import com.komputation.cuda.setIntArray
import com.komputation.cuda.setUpCudaContext
import jcuda.Pointer
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.util.*

class CudaHashingTest {

    @Test
    fun testComplete() {

        val random = Random()

        listOf(1 to 2, 2 to 2, 3 to 2).forEach { (size, maximum) ->

            test(IntArray(size) { random.nextInt(maximum) })

        }

    }

    @Test
    fun testIncomplete1() {

        test(intArrayOf(-1))

    }

    @Test
    fun testIncomplete2() {

        test(intArrayOf(0, -1))

    }

    @Test
    fun testIncomplete3() {

        test(intArrayOf(0, -1, 0))

    }

    private fun test(indices: IntArray) {

        val context = setUpCudaContext()

        val hashing = CudaHashing(
            indices.size,
            2,
            { context.createKernel(HashtableKernels.hash()) },
            { context.createKernel(ArrayKernels.fillTwoIntegerArrays()) },
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        hashing.acquire(1)

        val deviceIndices = Pointer()
        setIntArray(indices, indices.size, deviceIndices)

        hashing.hash(Pointer.to(deviceIndices))

        val mapping = getIntArray(hashing.getDeviceMapping(), indices.size)

        check(indices, mapping)

        context.destroy()

    }

    private fun check(indices: IntArray, mapping: IntArray) {

        indices
            .zip(mapping)
            .filter { (_, mapping) -> mapping != -1 }
            .groupBy({ (index, _) -> index }, { (_, mapping) -> mapping})
            .values
            .map { it.toSet() }
            .map { it.count() }
            .forEach { count ->

                assertTrue(count == 1)

            }

    }

}