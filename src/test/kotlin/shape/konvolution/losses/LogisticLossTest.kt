package shape.konvolution.losses

import org.junit.jupiter.api.Test
import shape.konvolution.loss.LogisticLoss
import org.junit.jupiter.api.Assertions.assertTrue
import shape.konvolution.assertMatrixEquality
import shape.konvolution.createRealMatrix

class LogisticLossTest {

    @Test
    fun testForward() {

        val logisticLoss = LogisticLoss()

        val targets = createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        )

        val prettyGoodPrediction = createRealMatrix(
            doubleArrayOf(0.3),
            doubleArrayOf(0.7)
        )

        val lossForPrettyGoodPrediction = logisticLoss.forward(
            prettyGoodPrediction,
            targets
        )

        val goodPrediction = createRealMatrix(
            doubleArrayOf(0.2),
            doubleArrayOf(0.8)
        )

        val lossForGoodPrediction = logisticLoss.forward(
            goodPrediction,
            targets
        )

        val veryGoodPrediction = createRealMatrix(
            doubleArrayOf(0.1),
            doubleArrayOf(0.9)
        )

        val lossForVeryGoodPrediction = logisticLoss.forward(
            veryGoodPrediction,
            targets
        )

        assertTrue(lossForGoodPrediction > lossForVeryGoodPrediction)
        assertTrue((lossForPrettyGoodPrediction - lossForGoodPrediction) > (lossForGoodPrediction - lossForVeryGoodPrediction))

    }

    @Test
    fun testBackward() {

        val logisticLoss = LogisticLoss()

        val targets = createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(1.0)
        )

        val prediction = createRealMatrix(
            doubleArrayOf(0.3),
            doubleArrayOf(0.7)
        )

        val actual = logisticLoss.backward(
            prediction,
            targets
        )
        val expected = createRealMatrix(
            doubleArrayOf(0.0),
            doubleArrayOf(-1.0.div(0.7))
        )

        assertMatrixEquality(expected, actual, 0.001)

    }

}