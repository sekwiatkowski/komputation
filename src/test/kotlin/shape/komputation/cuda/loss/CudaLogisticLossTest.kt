package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setFloatArray
import shape.komputation.loss.logisticLoss
import shape.komputation.matrix.FloatMath

class CudaLogisticLossTest {

    @Test
    fun testForwardTwoCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f)
        val targets = floatArrayOf(0.0f, 1.0f)

        val expected = -FloatMath.log(0.7f)

        testForward(predictions, targets, 2, 1, expected)

    }

    @Test
    fun testForwardThreeCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.5f, 0.3f, 0.2f)
        val targets = floatArrayOf(1.0f, 0.0f, 0.0f)

        val expected = -FloatMath.log(0.5f)

        testForward(predictions, targets, 3, 1, expected)

    }

    @Test
    fun testForwardTwoCategoriesTwoStepsDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f, 0.4f, 0.6f)
        val targets = floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f)

        val expected = -FloatMath.log(0.7f) - FloatMath.log(0.4f)

        testForward(predictions, targets, 2, 2, expected)

    }

    @Test
    fun testBackwardTwoCategoriesOneStepDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f)
        val targets = floatArrayOf(0.0f, 1.0f)

        val expected = floatArrayOf(0.0f, -1.0f/0.7f)

        testBackward(predictions, targets, 2, 1, expected)

    }

    @Test
    fun testBackwardTwoCategoriesTwoStepsDimension() {

        val predictions = floatArrayOf(0.3f, 0.7f, 0.8f, 0.2f)
        val targets = floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f)

        val expected = floatArrayOf(0.0f, -1.0f/0.7f, -1.0f/0.8f, 0.0f)

        testBackward(predictions, targets, 2, 2, expected)

    }


    private fun testForward(predictions: FloatArray, targets: FloatArray, numberCategories : Int, numberSteps : Int, expected: Float) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(cudaContext)

        loss.acquire()

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets))

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001f)

    }

    private fun testBackward(predictions: FloatArray, targets: FloatArray, numberCategories : Int, numberSteps : Int, expected: FloatArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setFloatArray(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setFloatArray(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(context)

        loss.acquire()

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets))
        val actual = getFloatArray(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        assertArrayEquals(expected, actual, 0.001f)

    }


}