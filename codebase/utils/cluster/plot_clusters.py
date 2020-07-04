import numpy as np

import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_directory(cluster_path, height=7, width=9, cropped=False):
    plot_name = "grid.pdf"
    if cropped:
        plot_name = "grid_cropped.pdf"

    images_counter = 0
    for cluster_index in os.listdir(cluster_path):
        print("---- Plotting directory: {} ----".format(cluster_index))
        current_cluster_directory = os.path.join(cluster_path, cluster_index)
        images_counter = 0

        if os.path.isdir(current_cluster_directory):
            fig, axes = plt.subplots(height, width, gridspec_kw={'hspace': 0, 'wspace': 0})
            for current_image in os.listdir(current_cluster_directory):
                current_image_path = os.path.join(current_cluster_directory, current_image)
                if ".png" in current_image and ("orig" in current_image) and ("tf1" in current_image):
                    if (("crop" in current_image) and cropped) or (("crop" not in current_image) and not cropped):
                        image = np.asarray(Image.open(current_image_path))
                        axes[images_counter // width][images_counter % width].imshow(image)
                        axes[images_counter // width][images_counter % width].axis("off")

                        images_counter += 1
                        if images_counter == width * height:
                            break

            if images_counter > 0:
                fig.savefig(os.path.join(cluster_path, "{}_{}".format(cluster_index, plot_name)), dpi=300, bbox_inches='tight')
                fig.clf()
                plt.close()

if __name__ == "__main__":

    cluster_path = "/media/veracrypt1/exact/thesis/iic/office_home/03_ensemble_learn_vanilla_align5-4/cluster_examples/5303/a"
    print("------ Plotting full image versions ------")
    plot_directory(cluster_path)
    print("------ Plotting cropped image versions ------")
    #plot_directory(cluster_path, cropped=True)
    print("------ Finished ------")
