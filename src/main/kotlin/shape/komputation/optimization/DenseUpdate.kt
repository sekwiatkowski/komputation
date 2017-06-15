package shape.komputation.optimization

fun updateDensely(parameterEntries: DoubleArray, gradientEntries : DoubleArray, rule : UpdateRule) {

    val numberParameterEntries = parameterEntries.size

    for (index in 0..numberParameterEntries - 1) {

        val current = parameterEntries[index]
        val derivative = gradientEntries[index]

        parameterEntries[index] = rule.apply(index, current, derivative)

    }

}