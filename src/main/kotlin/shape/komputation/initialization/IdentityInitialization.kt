package shape.komputation.initialization

fun createIdentityInitializer(): InitializationStrategy {

    return { indexRow : Int, indexColumn : Int ->

        if (indexRow == indexColumn) {
            1.0
        }
        else {
            0.0
        }

    }

}
