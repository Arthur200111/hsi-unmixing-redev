import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from module.dictionary import Dictionary

class MapGenerator:
    A: Dictionary
    element_indexes: list[int]
    
    width: int
    height: int
    snr: int
    prop: int
    
    pure_px_coords: list[tuple[int]]
    
    map_x: npt.NDArray[np.float64]
    map_y: npt.NDArray[np.float64]
    map_y_gt: npt.NDArray[np.float64]
    map_noise: npt.NDArray[np.float64]
    
    def __init__(self, dictionary: Dictionary, width: int = 50, height: int = 50, k: int = 5, snr: int = 40, prop = 2) -> None:
        self.A = dictionary
        self.width = width
        self.height = height
        self.element_indexes = self.A.get_random_elements(k)
        self.prop = prop
        self.snr = snr
    
    def compute_maps(self):
        self.map_x = np.zeros((self.width, self.height, len(self.element_indexes)))
        self.compute_map_x()
        
        self.map_y_gt = np.zeros((self.width, self.height, self.A.N))
        self.map_y = np.zeros((self.width, self.height, self.A.N))
        self.map_noise = np.zeros((self.width, self.height, self.A.N))
        self.compute_map_y()
        
    def compute_map_y2(self):
        cte = pow(10, self.snr/10)
        for i in range(self.width):
            for j in range(self.height):
                x = self.map_x[i, j]
                y_gt = self.A.speclib[:, self.element_indexes] @ x
                self.map_y_gt[i, j] = y_gt
                theta2 = np.inner(y_gt, y_gt)/cte
                noise = np.random.normal(0, theta2, self.A.N)
                self.map_noise[i, j] = noise
                y = y_gt + noise
                self.map_y[i, j] = y 
                 
    def compute_map_y(self):
        cte = pow(10, self.snr/10)
        x = self.map_x.reshape((-1, len(self.element_indexes)))
        y_gt = x @ self.A.speclib[:, self.element_indexes].T
        noise = np.zeros((len(x), len(self.A.speclib)))
        
        for n in range(len(x)):
            theta2 = np.inner(y_gt[n], y_gt[n])/cte
            noise[n] = np.random.normal(0, theta2, self.A.N)
            
        y = y_gt + noise
        
        self.map_y = y.reshape((self.height, self.width, len(self.A.speclib)))
        self.map_y_gt = y_gt.reshape((self.height, self.width, len(self.A.speclib)))
        self.map_noise = noise.reshape((self.height, self.width, len(self.A.speclib)))
    
    def compute_map_x(self):
        ## Place pure pixels
        pure_n = np.random.choice(self.element_indexes, size=len(self.element_indexes) * self.prop)
        self.pure_px_coords = np.zeros((len(pure_n), 2))
        coords = np.random.choice(self.height*self.width, replace=False, size=len(pure_n))
        
        for k in range(len(coords)):
            i = coords[k]//self.width
            j = coords[k] - self.width * i
            self.pure_px_coords[k] = np.array([int(i), int(j)])
        self.pure_px_coords = self.pure_px_coords.astype(int)
            
        for y in range(self.height):
            for x in range(self.width):
                if True in np.all(self.pure_px_coords == [x, y], axis = 1):
                    continue
                pixel = self.map_x[x,y]
                for k in range(len(self.pure_px_coords)):
                    coord = self.pure_px_coords[k]
                    elemIndex = np.where(pure_n[k] == self.element_indexes)
                    d = 1/((x - coord[0])**2 + (y - coord[1])**2)
                    pixel[elemIndex] += d
                pixel /= sum(pixel)
    
    def plot_x(self, map: npt.NDArray[np.float64] = None):
        if map is None: map = self.map_x
        
        n = len(self.element_indexes)
            
        fig, axes = plt.subplots(nrows=2, ncols=n//2+1, figsize=(12, 8))

        for k in range(np.prod(axes.shape)):
            ax = axes.flat[k]
            ax.axis("off")
            ax.get_yaxis().set_visible(False)
            if k >= n:
                continue
            i = self.element_indexes[k]
            ax.title.set_text(f"{self.A.mineral_names[i]}")
            im = ax.imshow(map[:, :, k], vmin=0, vmax=1)

        fig.subplots_adjust(right=0.8) 
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    def plot_y(self, coord: tuple[int]= (0,0)):
        plt.plot(self.A.wavelength, self.map_y_gt[coord], label="A*X_xy")
        plt.plot(self.A.wavelength, self.map_y[coord], label="Y_xy")
        plt.legend()
        plt.title("Comparaison between Y_GT (=A*X) and Y for a chosen pixel")
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Amplitude")
