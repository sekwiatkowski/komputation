package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector
import shape.komputation.loss.logisticLoss

class CudaLogisticLossTest {

    @Test
    fun testForwardTwoCategoriesOneStepDimension() {

        val predictions = doubleArrayOf(0.3, 0.7)
        val targets = doubleArrayOf(0.0, 1.0)

        val expected = -Math.log(0.7)

        testForward(predictions, targets, 2, 1, expected)

    }

    @Test
    fun testForwardThreeCategoriesOneStepDimension() {

        val predictions = doubleArrayOf(0.5, 0.3, 0.2)
        val targets = doubleArrayOf(1.0, 0.0, 0.0)

        val expected = -Math.log(0.5)

        testForward(predictions, targets, 3, 1, expected)

    }

    @Test
    fun testForwardTwoCategoriesTwoStepsDimension() {

        val predictions = doubleArrayOf(0.3, 0.7, 0.4, 0.6)
        val targets = doubleArrayOf(0.0, 1.0, 1.0, 0.0)

        val expected = -Math.log(0.7) - Math.log(0.4)

        testForward(predictions, targets, 2, 2, expected)

    }

    @Test
    fun testBackwardTwoCategoriesOneStepDimension() {

        val predictions = doubleArrayOf(0.3, 0.7)
        val targets = doubleArrayOf(0.0, 1.0)

        val expected = doubleArrayOf(0.0, -1.0/0.7)

        testBackward(predictions, targets, 2, 1, expected)

    }

    @Test
    fun testBackwardTwoCategoriesTwoStepsDimension() {

        val predictions = doubleArrayOf(0.3, 0.7, 0.8, 0.2)
        val targets = doubleArrayOf(0.0, 1.0, 1.0, 0.0)

        val expected = doubleArrayOf(0.0, -1.0/0.7, -1.0/0.8, 0.0)

        testBackward(predictions, targets, 2, 2, expected)

    }


    private fun testForward(predictions: DoubleArray, targets: DoubleArray, numberCategories : Int, numberSteps : Int, expected: Double) {

        val size = predictions.size

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(cudaContext)

        loss.acquire()

        loss.accumulate(Pointer.to(devicePredictions), Pointer.to(deviceTargets))

        val actual = loss.accessAccumulation()

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        cudaContext.destroy()

        assertEquals(expected, actual, 0.001)

    }

    private fun testBackward(predictions: DoubleArray, targets: DoubleArray, numberCategories : Int, numberSteps : Int, expected: DoubleArray) {

        val size = predictions.size

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val loss = logisticLoss(numberCategories, numberSteps).buildForCuda(context)

        loss.acquire()

        val deviceResult = loss.backward(Pointer.to(devicePredictions), Pointer.to(deviceTargets))
        val actual = getVector(deviceResult, size)

        loss.release()

        cudaFree(devicePredictions)
        cudaFree(deviceTargets)

        context.destroy()

        Assertions.assertArrayEquals(expected, actual, 0.001)

    }


}