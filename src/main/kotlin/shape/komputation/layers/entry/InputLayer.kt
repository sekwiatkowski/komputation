package shape.komputation.layers.entry

import shape.komputation.matrix.Matrix
import shape.komputation.matrix.RealMatrix

class InputLayer(name : String? = null) : EntryPoint(name) {

    override fun forward(input : Matrix) : RealMatrix {

        return input as RealMatrix

    }

}