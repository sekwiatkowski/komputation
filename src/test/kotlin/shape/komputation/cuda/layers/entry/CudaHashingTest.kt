package shape.komputation.cuda.layers.entry

import jcuda.Pointer
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getIntArray
import shape.komputation.cuda.setIntArray
import shape.komputation.cuda.setUpCudaContext
import java.util.*
import org.junit.jupiter.api.Assertions.assertTrue
import shape.komputation.cuda.kernels.FillKernels
import shape.komputation.cuda.kernels.HashtableKernels

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
            { context.createKernel(FillKernels.twoIntegerArrays()) })

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