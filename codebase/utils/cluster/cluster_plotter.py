import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid


class ClusterPlotter:

    @staticmethod
    def plot_all(clustering_information, height, width, basename, output_path, maxplots=15):

        # Created directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        images_per_grid = height * width

        for predicted_label in clustering_information:
            current_information = clustering_information[predicted_label]
            random.shuffle(current_information)

            current_plot_index = 0
            current_image_index = 0
            num_images = len(current_information)
            while current_image_index < num_images and current_plot_index < maxplots:
                fig = plt.figure()
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(height, width),
                                 axes_pad=0.0,  # pad between axes in inch.
                                 )

                for ax in grid:
                    if current_image_index < num_images:

                        image_path = current_information[current_image_index][0]
                        image = mpimg.imread(image_path)

                        ax.imshow(image, interpolation="none")
                        color = "green"
                        if current_information[current_image_index][1] != current_information[current_image_index][2]:
                            color = "red"
                        #ax.spines["bottom"].set_color("green")
                        ax.plot((0, 0), (0, 149), color=color, linewidth=5)
                        ax.plot((0, 149), (0, 0), color=color, linewidth=5)
                        ax.plot((0, 149), (149, 149), color=color, linewidth=5)
                        ax.plot((149, 149), (0, 149), color=color, linewidth=5)
                        current_image_index += 1
                    ax.axis('off')


                #fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
                figure_name = "{}_{}_{}.pdf".format(basename, predicted_label, current_plot_index)
                print("Saving image grid: {}".format(figure_name))
                fig.savefig(os.path.join(output_path, figure_name), dpi=150, bbox_inches='tight')

                plt.close(fig)
                current_plot_index += 1
