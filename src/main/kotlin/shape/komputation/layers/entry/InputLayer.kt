package shape.komputation.layers.entry

import shape.komputation.matrix.RealMatrix

class InputLayer(name : String? = null) : EntryPoint(name) {

    override fun forward() {

        this.lastForwardResult = this.lastInput!! as RealMatrix

    }

}