package com.komputation.cuda.kernels.launch

fun computeNumberSegments(totalEntries: Int, entriesPerSegment: Int) =
    (totalEntries + entriesPerSegment - 1) / entriesPerSegment