def integral(bins, edges):
    area = 0.0
    for i in range(len(edges) - 1):
        width = edges[i+1] - edges[i]
        height = (bins[i+1] + bins[i]) / 2
        area += width * height
    return area