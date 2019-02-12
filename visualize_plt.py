import numpy as np
import matplotlib.pyplot as plt

class Window:

    def __init__(self):
        super(Window, self).__init__()

    def plot_loss(self, losses, title):
        epochs = np.arrange(1, len(losses)+1, 1)
        plt.plot(epochs, losses)
        plt.xticks(rotation=45)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.show()

    def show_image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        image = image.numpy()

        plt.figure()
        plt.title(title)
        plt.imshow(image)
        plt.show()

