import numpy as np

def convert2D(M: np.ndarray):
    """
    M(h, w, p) => M(p,N) with N = h*w
    """
    ndim = len(M.size)
    if ndim < 2 or ndim > 3:
        raise ValueError("Input image must be m x n x p or m x n")
    if (ndim == 2):
        p = 1
        h, w = M.shape
    else:
        h, w, p = M.shape
    return M.reshape((w*h, p)).T

def convert3D(M, h, w, p):
    """
    M(p,N) => M(h, w, p)
    """
    ndim = len(M.size)
    if ndim != 2:
        raise ValueError("Input image must be p x N.")
    
    return M.T.reshape((h, w, p))

def getPatches(image, n = 16):
    h, w, d = image.shape
    margin = n//4
    k_w = (w + 1) // (3 * margin)
    k_h = (h + 1) // (3 * margin)
    
    keys = [[[] for _ in range(w)] for _ in range(h)]
    
    patches = np.zeros((k_w * k_h, n, n, d))
    
    for x in range(k_w):
        for y in range(k_h):
            x_coord = x * 3 * margin
            if x == k_w - 1: x_coord = w - n
            
            y_coord = y * 3 * margin
            if y == k_h - 1: y_coord = h - n
            
            for i in range(n):
                for j in range(n):
                    keys[y_coord + j][x_coord + i].append((x+y*k_w, j, i))
            
            patches[x + y * k_w] = image[y_coord: y_coord + n, x_coord: x_coord + n]
    return patches, keys

def getImageFromPatches(patches, keys):
    h, w = len(keys), len(keys[0])
    d = len(patches[0, 0, 0])
    
    image = np.zeros((h, w, d))
    for x in range(w):
        for y in range(h):
            s = np.zeros(d)
            for coord in keys[y][x]:
                s += patches[coord]
            image[y, x] = s / len(keys[y][x])
    return image