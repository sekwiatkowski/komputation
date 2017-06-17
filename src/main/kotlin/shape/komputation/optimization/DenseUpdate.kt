package shape.komputation.optimization

fun updateDensely(parameterEntries: DoubleArray, gradientEntries : DoubleArray, batchSize : Int, rule : UpdateRule) {

    val numberParameterEntries = parameterEntries.size
    val scalingFactor = 1.0.div(batchSize.toDouble())

    for (index in 0..numberParameterEntries - 1) {

        val current = parameterEntries[index]
        val derivative = scalingFactor * gradientEntries[index]

        parameterEntries[index] = rule.apply(index, current, derivative)

    }

}