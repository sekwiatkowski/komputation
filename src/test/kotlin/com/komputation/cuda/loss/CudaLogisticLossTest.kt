package com.komputation.cuda.loss

import com.komputation.cuda.getFloatArray
import com.komputation.cuda.setFloatArray
import com.komputation.cuda.setUpCudaContext
import com.komputation.loss.logisticLoss
import com.komputation.matrix.FloatMath
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class CudaLogisticLossTest {

    @Test
    fun testForwardPositiveOneDimensionInOneInstance() {

        val predictions = floatArrayOf(0.6f)
        val targets = floatArrayOf(1f)

        val expected = -FloatMath.log(0.6f)

        testForward(predictions, targets, 1, 1, 1, expected)

    }

    @Test
    fun testForwardNegativeOneDimensionInOneInstance() {

        val predictions = floatArrayOf(0.6f)
        val targets = floatArrayOf(0f)

        val expected = -FloatMath.log(0.4f)

        testForward(predictions, targets, 1, 1, 1, expected)

    }

    @Test
    fun testForwardTwoDimensionsInOneInstance() {

        val predictions = floatArrayOf(0.8f, 0.3f)
        val targets = floatArrayOf(1f, 0f)

        val expected = -FloatMath.log(0.8f * 0.7f)

        testForward(predictions, targets, 1, 1, 2, expected)

    }

    @Test
    fun testForwardOneDimensionInTwoInstances() {

        val predictions = floatArrayOf(0.8f, 0.3f)
        val targets = floatArrayOf(1f, 0f)

        val expected = -FloatMath.log(0.8f * 0.7f)

        testForward(predictions, targets, 2, 2, 1, expected)

    }

    @Test
    fun testForwardTwoDimensionsInTwoInstances() {

        val predictions = floatArrayOf(0.8f, 0.3f, 0.4f, 0.9f)
        val targets = floatArrayOf(1f, 0f, 0f, 1f)

        val expected = -FloatMath.log(0.8f * 0.7f * 0.6f * 0.9f)

        testForward(predictions, targets, 2, 2, 2, expected)

    }

    @Test
    fun testForwardIncompleteBatch() {

        val predictions = floatArrayOf(0.8f, Float.NaN)
        val targets = floatArrayOf(1f, Float.NaN)

        val expected = -FloatMath.log(0.8f)

        testForward(predictions, targets, 1, 2, 1, expected)

    }

    private fun testForward(predictions: FloatArray, targets: FloatArray, batchSize: Int, maximumBatchSize: Int, numberSteps: Int, expected: Float) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberSteps).buildForCuda(cudaContext)

        loss.acquire(maximumBatchSize)

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets), batchSize)

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001f)

    }

    @Test
    fun testBackwardPositiveOneDimension() {

        val predictions = floatArrayOf(0.9f)
        val targets = floatArrayOf(1f)

        val expected = floatArrayOf(-1.0f/0.9f)

        testBackward(predictions, targets, 1, 1, 1, expected)

    }

    @Test
    fun testBackwardNegativeOneDimension() {

        val predictions = floatArrayOf(0.2f)
        val targets = floatArrayOf(0f)

        val expected = floatArrayOf(1.0f/0.8f)

        testBackward(predictions, targets, 1, 1, 1, expected)

    }

    @Test
    fun testBackwardTwoDimensionsInOneInstance() {

        val predictions = floatArrayOf(0.9f, 0.2f)
        val targets = floatArrayOf(1f, 0f)

        val expected = floatArrayOf(-1.0f/0.9f, 1.0f/0.8f)

        testBackward(predictions, targets, 1, 2, 2, expected)

    }

    @Test
    fun testBackwardOneDimensionInTwoInstances() {

        val predictions = floatArrayOf(0.9f, 0.2f)
        val targets = floatArrayOf(1f, 0f)

        val expected = floatArrayOf(-1.0f/0.9f, 1.0f/0.8f)

        testBackward(predictions, targets, 2, 2, 1, expected)

    }

    @Test
    fun testBackwardIncompleteBatch() {

        val predictions = floatArrayOf(0.8f, Float.NaN)
        val targets = floatArrayOf(1f, Float.NaN)

        val expected = floatArrayOf(-1.0f/0.8f, Float.NaN)

        testBackward(predictions, targets, 1, 2, 1, expected)

    }

    private fun testBackward(predictions: FloatArray, targets: FloatArray, batchSize: Int, maximumBatchSize: Int, numberSteps: Int, expected: FloatArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberSteps).buildForCuda(context)

        loss.acquire(maximumBatchSize)

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets), batchSize)
        val actual = getFloatArray(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}