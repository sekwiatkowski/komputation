package shape.komputation.initialization

class IdentityInitialization internal constructor(): InitializationStrategy {

    override fun initialize(indexRow: Int, indexColumn: Int, numberIncoming: Int) =

        if (indexRow == indexColumn) {
            1.0
        }
        else {
            0.0
        }

}

fun identityInitialization() = IdentityInitialization()