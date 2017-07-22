package shape.komputation.cpu.functions.activation

fun relu(input: FloatArray) =

    FloatArray(input.size) { index ->

        relu(input[index])

    }

fun relu(entry: Float) =

    Math.max(entry, 0.0f)

// d relu(x) / d x = d max(x, 0) / d x
// = 1 if x > 0.0, 0.0 otherwise
fun backwardRelu(forwardEntries : FloatArray, chainEntries : FloatArray) =

    FloatArray(forwardEntries.size) { index ->

        val forwardEntry = forwardEntries[index]

        if (forwardEntry > 0.0)
            chainEntries[index]
        else
            0.0f

    }