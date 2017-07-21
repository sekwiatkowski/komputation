package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.loss.squaredLoss

class CudaSquaredLossTest {

    @Test
    fun testForwardOneDimension() {

        val expected = 0.5 * Math.pow(4.0-2.0, 2.0)

        val predictions = doubleArrayOf(4.0)
        val targets = doubleArrayOf(2.0)

        testForward(predictions, targets, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val expected = 0.5 * (Math.pow(4.0-2.0, 2.0) + Math.pow(6.0-3.0, 2.0))

        val predictions = doubleArrayOf(4.0, 6.0)
        val targets = doubleArrayOf(2.0, 3.0)

        testForward(predictions, targets, expected)

    }

    @Test
    fun testForwardThreeDimensions() {

        val expected = 0.5 * (Math.pow(4.0-2.0, 2.0) + Math.pow(6.0-3.0, 2.0) + Math.pow(8.0-4.0, 2.0))

        val predictions = doubleArrayOf(4.0, 6.0, 8.0)
        val targets = doubleArrayOf(2.0, 3.0, 4.0)

        testForward(predictions, targets, expected)

    }


    private fun testForward(predictions: DoubleArray, targets: DoubleArray, expected: Double) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val loss = squaredLoss(size).buildForCuda(cudaContext)

        loss.acquire()

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets))

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001)

    }

    @Test
    fun testBackwardOneDimension() {

        val expected = doubleArrayOf(2.0)

        val predictions = doubleArrayOf(4.0)
        val targets = doubleArrayOf(2.0)

        testBackward(predictions, targets, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val expected = doubleArrayOf(2.0, 3.0)

        val predictions = doubleArrayOf(4.0, 6.0)
        val targets = doubleArrayOf(2.0, 3.0)

        testBackward(predictions, targets, expected)

    }

    private fun testBackward(predictions: DoubleArray, targets: DoubleArray, expected: DoubleArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val loss = squaredLoss(size).buildForCuda(context)

        loss.acquire()

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets))
        val actual = getVector(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001)

    }


}