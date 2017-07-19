package shape.komputation.cuda.loss

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.KernelFactory
import shape.komputation.cuda.getVector
import shape.komputation.cuda.setUpCudaContext
import shape.komputation.cuda.setVector

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

        val cudaContext = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)
        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val forwardKernel = KernelFactory(cudaContext.computeCapabilities)

        val loss = CudaSquaredLoss(
            forwardKernel.squaredLoss(),
            forwardKernel.backwardSquaredLoss(),
            size)

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

        val context = setUpCudaContext()

        val devicePredictions = Pointer()
        setVector(predictions, size, devicePredictions)

        val deviceTargets = Pointer()
        setVector(targets, size, deviceTargets)

        val kernelFactory = KernelFactory(context.computeCapabilities)

        val loss = CudaSquaredLoss(
            kernelFactory.squaredLoss(),
            kernelFactory.backwardSquaredLoss(),
            size)

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