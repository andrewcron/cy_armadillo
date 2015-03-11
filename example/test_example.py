import numpy as np
import example
from six import print_

if __name__ == "__main__":
    x = np.random.rand(20).reshape(5,4)

    print_(example.example(x))

