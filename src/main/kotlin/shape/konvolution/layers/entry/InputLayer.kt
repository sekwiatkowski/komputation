package shape.konvolution.layers.entry

import shape.konvolution.matrix.RealMatrix

class InputLayer(name : String? = null) : EntryPoint(name) {

    override fun forward() {

        this.lastForwardResult = this.lastInput!! as RealMatrix

    }

}