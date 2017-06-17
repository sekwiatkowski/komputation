package shape.komputation.functions.activation

fun relu(input: DoubleArray) =

    DoubleArray(input.size) { index ->

        relu(input[index])

    }

fun relu(entry: Double) =

    Math.max(entry, 0.0)

// d relu(x) / d x = d max(x, 0) / d x
// = 1 if x > 0.0, 0.0 otherwise
fun backwardRelu(forwardEntries : DoubleArray, chainEntries : DoubleArray) =

    DoubleArray(forwardEntries.size) { index ->

        val forwardEntry = forwardEntries[index]

        if (forwardEntry > 0.0)
            chainEntries[index]
        else
            0.0

    }