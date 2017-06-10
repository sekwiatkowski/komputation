package shape.konvolution.layers.continuation

/* class Concatenation(private val dimension: Int, vararg layerSequences: Array<ContinuationLayer>) : ContinuationLayer {

    val networks = layerSequences.map { layers -> Network(InputLayer(), *layers) }

    val numberNetworks = networks.size

    override fun forward(input: RealMatrix): RealMatrix {

        val branches = Array(numberNetworks) { indexBranch ->

            networks[indexBranch].forward(input).last()

        }

        val concatenation = concatColumns(branches, numberNetworks, 1, dimension)

        return concatenation

    }

    override fun backward(input: RealMatrix, output : RealMatrix, chain : RealMatrix): BackwardResult {

        TODO("not implemented")
    }

} */