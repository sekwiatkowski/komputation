package shape.komputation.cuda.layers.forward.expansion

import jcuda.Pointer
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.kernels.ForwardKernels
import shape.komputation.cuda.setFloatArray
import shape.komputation.cuda.setIntArray
import shape.komputation.cuda.setUpCudaContext

class CudaExpansionLayerTest {

    @Test
    fun testOneByOneConvolutionOneByOneInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f)
        val expected = floatArrayOf(1f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 1, input, expected)

    }

    @Test
    fun testIncompleteBatch() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN)
        val expected = floatArrayOf(1f, Float.NaN)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 2, 1, 1, input, expected)

    }

    @Test
    fun testOneByOneConvolutionTwoByOneInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, 2f)
        val expected = floatArrayOf(1f, 2f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 1, input, expected)

    }

    @Test
    fun testOneByOneConvolutionTwoByTwoInput() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 2, input, expected)

    }

    @Test
    fun testTwoByTwoConvolutionTwoByTwoInput() {

        val filterHeight = 2
        val filterWidth = 2
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val expected = floatArrayOf(1f, 2f, 3f, 4f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 2, 2, input, expected)

    }

    @Test
    fun testTwoByTwoConvolutionThreeByTwoThree() {

        /*
            1 4 7
            2 5 8
            3 6 9

            1, 2, 4, 5
            2, 5, 3, 6
            4, 5, 6, 7
            5, 7, 6, 9

         */

        val filterHeight = 2
        val filterWidth = 2
        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val expected = floatArrayOf(
            1f, 2f, 4f, 5f,
            2f, 3f, 5f, 6f,
            4f, 5f, 7f, 8f,
            5f, 6f, 8f, 9f)

        testFixedLengthForwardPropagation(filterHeight, filterWidth, 1, 3, 3, input, expected)

    }

    private fun testFixedLengthForwardPropagation(filterHeight: Int, filterWidth: Int, maximumBatchSize : Int, numberInputRows: Int, maximumInputColumns : Int, input: FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            numberInputRows,
            maximumInputColumns,
            filterHeight,
            filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        expansionLayer.acquire(maximumBatchSize)
        val deviceResult = expansionLayer.forward(1, deviceInput, false)

        val actual = getFloatArray(deviceResult, maximumBatchSize * expansionLayer.maximumOutputColumns * expansionLayer.numberOutputRows)

        context.destroy()

        assertArrayEquals(expected, actual, 0.01f)

    }

    @Test
    fun testOneOutOfTwoColumnsInstance() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN)
        val expected = floatArrayOf(1f, 0f)

        testVariableLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 2, intArrayOf(1), input, expected)

    }

    @Test
    fun testOneOutOfThreeColumnsInstance() {

        val filterHeight = 1
        val filterWidth = 1
        val input = floatArrayOf(1f, Float.NaN, Float.NaN)
        val expected = floatArrayOf(1f, 0f, 0f)

        testVariableLengthForwardPropagation(filterHeight, filterWidth, 1, 1, 3, intArrayOf(1), input, expected)

    }

    private fun testVariableLengthForwardPropagation(filterHeight: Int, filterWidth: Int, maximumBatchSize : Int, numberInputRows: Int, maximumInputColumns : Int, lengths : IntArray, input: FloatArray, expected: FloatArray) {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            numberInputRows,
            maximumInputColumns,
            filterHeight,
            filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        val deviceLengths = Pointer()
        setIntArray(lengths, lengths.size, deviceLengths)

        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        expansionLayer.acquire(maximumBatchSize)
        val deviceResult = expansionLayer.forward(1, deviceLengths, deviceInput, false)

        val actual = getFloatArray(deviceResult, maximumBatchSize * expansionLayer.maximumOutputColumns * expansionLayer.numberOutputRows)

        expansionLayer.release()

        context.destroy()

        assertArrayEquals(expected, actual, 0.01f)
    }

    @Test
    fun testBackward() {

        val context = setUpCudaContext()

        val expansionLayer = CudaExpansionLayer(
            null,
            1,
            1,
            1,
            1,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

        expansionLayer.acquire(1)

        val input = floatArrayOf(1f)
        val deviceInput = Pointer()
        setFloatArray(input, input.size, deviceInput)

        val chain = floatArrayOf(1f)
        val deviceChain = Pointer()
        setFloatArray(chain, chain.size, deviceChain)

        expansionLayer.forward(1, deviceInput, false)
        expansionLayer.backward(1, deviceChain)

        expansionLayer.release()

        context.destroy()

    }


}