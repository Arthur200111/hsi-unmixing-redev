import numpy as np
from scipy.io import loadmat
import random as rd
import matplotlib.pyplot as plt

class Dictionary:
    speclib: np.ndarray
    mineral_names: list = [] #names of minerals
    wavelength: np.ndarray #wavelenghts
    N: int #
    P: int
    
    def __init__(self, path: str, n_elements=-1):
        dictionary: dict = loadmat(path)
        self.speclib = dictionary["speclib"]
        for names_list in np.array(dictionary["list_spec_in_mineral"][0]):
            for names in list(names_list)[2]:
                for name in names:
                    self.mineral_names.append(str(name[0]))
        self.wavelength = dictionary["wavelength"][:,0]
        self.N, self.P = self.speclib.shape
        if n_elements > 0:
            a = self.get_random_elements(n_elements)
            self.speclib = self.speclib[:,a]
            self.mineral_names = [self.mineral_names[k] for k in a]
            self.P = n_elements
    
    def get_random_elements(self, k: int) -> list:
        return np.sort(np.random.choice(self.P, replace=False, size=k))
    
    def plot(self, indexes: list[int]):
        for index in np.unique(indexes):
            plt.plot(self.wavelength, self.speclib[:, index], label=self.mineral_names[index])
        plt.title("Spectra of the selected endmembers")
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Amplitude")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()