package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.copyFromHostToDevice
import shape.komputation.cuda.getVector

class CudaSquaredLossTest {

    @Test
    fun testForwardOneDimension() {

        val expected = 0.5 * Math.pow(4.0-2.0, 2.0)

        val predictions = doubleArrayOf(4.0)
        val targets = doubleArrayOf(2.0)
        val size = 1

        testForward(predictions, targets, size, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val expected = 0.5 * (Math.pow(4.0-2.0, 2.0) + Math.pow(6.0-3.0, 2.0))

        val predictions = doubleArrayOf(4.0, 6.0)
        val targets = doubleArrayOf(2.0, 3.0)
        val size = 2

        testForward(predictions, targets, size, expected)

    }

    private fun testForward(predictions: DoubleArray, targets: DoubleArray, size: Int, expected: Double) {

        val devicePredictions = Pointer()
        copyFromHostToDevice(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        copyFromHostToDevice(targets, size, deviceTargets)

        val loss = CudaSquaredLoss(3 to 5, size)

        loss.acquire()

        loss.accumulate(devicePredictions, deviceTargets)

        val actual = loss.accessAccumulation()

        loss.release()

        JCuda.cudaFree(devicePredictions)
        JCuda.cudaFree(deviceTargets)

        assertEquals(expected, actual, 0.001)
    }

    @Test
    fun testBackwardOneDimension() {

        val expected = doubleArrayOf(2.0)

        val predictions = doubleArrayOf(4.0)
        val targets = doubleArrayOf(2.0)
        val size = 1

        testBackward(predictions, targets, size, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val expected = doubleArrayOf(2.0, 3.0)

        val predictions = doubleArrayOf(4.0, 6.0)
        val targets = doubleArrayOf(2.0, 3.0)
        val size = 2

        testBackward(predictions, targets, size, expected)

    }

    private fun testBackward(predictions: DoubleArray, targets: DoubleArray, size: Int, expected: DoubleArray) {

        val devicePredictions = Pointer()
        copyFromHostToDevice(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        copyFromHostToDevice(targets, size, deviceTargets)

        val loss = CudaSquaredLoss(3 to 5, size)

        loss.acquire()

        val deviceResult = loss.backward(devicePredictions, deviceTargets)
        val actual = getVector(deviceResult, size)

        loss.release()

        JCuda.cudaFree(devicePredictions)
        JCuda.cudaFree(deviceTargets)

        assertArrayEquals(expected, actual, 0.001)

    }


}