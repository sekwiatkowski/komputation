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
import shape.komputation.matrix.FloatMath

class CudaSquaredLossTest {

    @Test
    fun testForwardOneDimension() {

        val expected = 0.5f * FloatMath.pow(4.0f-2.0f, 2.0f)

        val predictions = floatArrayOf(4.0f)
        val targets = floatArrayOf(2.0f)

        testForward(predictions, targets, expected)

    }


    @Test
    fun testForwardTwoDimensions() {

        val expected = 0.5f * (FloatMath.pow(4.0f-2.0f, 2.0f) + FloatMath.pow(6.0f-3.0f, 2.0f))

        val predictions = floatArrayOf(4.0f, 6.0f)
        val targets = floatArrayOf(2.0f, 3.0f)

        testForward(predictions, targets, expected)

    }

    @Test
    fun testForwardThreeDimensions() {

        val expected = 0.5f * (FloatMath.pow(4.0f-2.0f, 2.0f) + FloatMath.pow(6.0f-3.0f, 2.0f) + FloatMath.pow(8.0f-4.0f, 2.0f))

        val predictions = floatArrayOf(4.0f, 6.0f, 8.0f)
        val targets = floatArrayOf(2.0f, 3.0f, 4.0f)

        testForward(predictions, targets, expected)

    }


    private fun testForward(predictions: FloatArray, targets: FloatArray, expected: Float) {

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

        assertEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackwardOneDimension() {

        val expected = floatArrayOf(2.0f)

        val predictions = floatArrayOf(4.0f)
        val targets = floatArrayOf(2.0f)

        testBackward(predictions, targets, expected)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val expected = floatArrayOf(2.0f, 3.0f)

        val predictions = floatArrayOf(4.0f, 6.0f)
        val targets = floatArrayOf(2.0f, 3.0f)

        testBackward(predictions, targets, expected)

    }

    private fun testBackward(predictions: FloatArray, targets: FloatArray, expected: FloatArray) {

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

        assertArrayEquals(expected, actual, 0.001f)

    }


}