def integral(bins, edges):
    total_area = 0
    for i in range(len(bins) - 1):
        width = edges[i+1] - edges[i]
        height = bins[i] + bins[i+1]
        total_area += width * height / 2
    return total_area