def integral(bins, edges):
    area = 0
    for i in range(len(bins) - 1):
        width = edges[i + 1] - edges[i]
        height_avg = (bins[i] + bins[i + 1]) / 2
        area += width * height_avg
    return area