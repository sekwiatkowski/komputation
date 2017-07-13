package shape.komputation.cpu.forward.activation

/* class CudaSigmoidLayerTest {

    @Test
    fun testForwardOneDimension() {

        val environment = setUpCudaEnvironment()

        val layer = cudaSigmoidLayer(environment, 1)
        layer.acquire()

        val actual = layer.forward(doubleScalar(0.0), false)
        val expected = doubleScalar(0.5)

        layer.release()

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testForwardTwoDimensions() {

        val environment = setUpCudaEnvironment()

        val layer = cudaSigmoidLayer(environment, 2)
        layer.acquire()

        val actual = layer.forward(doubleRowVector(0.0, 1.0), false)
        val expected = doubleColumnVector(0.5, 0.731058579)

        layer.release()

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testBackwardOneDimension() {

        val environment = setUpCudaEnvironment()

        val layer = cudaSigmoidLayer(environment, 1)
        layer.acquire()

        layer.forward(doubleScalar(0.0), false)
        val actual = layer.backward(doubleScalar(1.0))

        layer.release()

        val expected = doubleScalar(0.25)

        assertMatrixEquality(expected, actual, 0.001)

    }

    @Test
    fun testBackwardTwoDimensions() {

        val environment = setUpCudaEnvironment()

        val layer = cudaSigmoidLayer(environment, 2)
        layer.acquire()

        layer.forward(doubleColumnVector(0.0, 1.0), false)
        val actual = layer.backward(doubleColumnVector(1.0, 2.0))

        layer.release()

        val expected = doubleColumnVector(1 * 0.5 * (1 - 0.5), 2 * sigmoid(1.0) * (1.0 - sigmoid(1.0)))

        assertMatrixEquality(expected, actual, 0.001)

    }


} */