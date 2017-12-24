package com.komputation.cuda.layers

import com.komputation.cuda.CudaBackwardPropagation
import com.komputation.cuda.CudaForwardPropagation

interface CudaContinuation : CudaForwardPropagation, CudaBackwardPropagation