import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button, TextBox
from matplotlib import path

class InteractiveDimensionReductionRegionsSelector:
    def __init__(self, data, embedding, y_pixels, x_pixels):
        self.data = data
        self.gx = x_pixels
        self.gy = y_pixels
        self.embedding = embedding
        self.img = np.zeros((np.amax(self.gy)+1, np.amax(self.gx)+1))
        self.ax1_xmin = np.amin(self.embedding[:,0]) - 1
        self.ax1_ymin = np.amin(self.embedding[:,1]) - 1
        self.ax1_xmax = np.amax(self.embedding[:,0]) + 1
        self.ax1_ymax = np.amax(self.embedding[:,1]) + 1
        self.toggle = "roi"
        self.toggle_props = {
            "roi": {'color': 'red', 'linewidth': 2, 'alpha': 0.8}
            }
        self.idx = {
            "reset": [],
            "roi": [],
            }
        self.rois = []


    def create_baseplot(self):
        self.fig = plt.figure()
        self.fig.suptitle("Interactive Dimension Reduction RoI Selection")
        self.ax1 = plt.axes([0.12, 0.2, 0.4, 0.75])
        self.ax1.set_title("Embedding")
        self.ax1.set_xlabel("Dimension A")
        self.ax1.set_ylabel("Dimension B")
        self.ax1.set_xlim([self.ax1_xmin, self.ax1_xmax])
        self.ax1.set_ylim([self.ax1_ymin, self.ax1_ymax])
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.ax1.set_aspect('equal')    
        
        self.ax2 = plt.axes([0.55, 0.2, 0.4, 0.75])
        self.ax2.axis("off")
        self.ax2.set_title('RoI Area:')
        self.ax2.imshow(self.img, vmax=1)
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props[self.toggle])
        plt.subplots_adjust(bottom=0.1)

        self.ax3 = plt.axes([0.25, 0.05, 0.12, 0.065])
        self.resetbutton = Button(self.ax3, "Reset")
        self.resetbutton.on_clicked(self.reset)

        self.ax4 = plt.axes([0.45, 0.05, 0.12, 0.065])
        self.roibutton = Button(self.ax4, "Set RoI")
        self.roibutton.on_clicked(self.exportroi)

        self.ax5 = plt.axes([0.65, 0.05, 0.22, 0.065])
        self.returnbutton = Button(self.ax5, "Return and Close")
        self.returnbutton.on_clicked(self.return_rois)

        
        

    def calc_img(self):
        self.img[:] = 0
        self.img[(self.gy[self.idx[self.toggle]], self.gx[self.idx[self.toggle]])] = 1
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.imshow(self.img, vmax=1)


    def onselect(self, verts):
        p = path.Path(verts)
        self.idx[self.toggle] = p.contains_points(self.pts.get_offsets())
        self.pts.remove()
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        self.ax1.scatter(self.embedding[self.idx["reset"], 0], self.embedding[self.idx["reset"], 1], s=6, c=self.toggle_props["roi"]["color"])
        self.ax1.scatter(self.embedding[self.idx["roi"], 0], self.embedding[self.idx["roi"], 1], s=6, c=self.toggle_props["roi"]["color"])
        self.calc_img()
        self.fig.canvas.draw_idle()


    def plot(self):
        self.create_baseplot()
        plt.show()
        

    def exportroi(self, event):
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props["roi"])
        self.idx["reset"] = []
        if not (self.img == 0).all():
            self.rois.append(self.img.copy())
        self.adjust_variables()
        self.adjust_plots()

    def reset(self, event):
        self.lsso = LassoSelector(ax=self.ax1, onselect=self.onselect, lineprops=self.toggle_props["roi"])
        self.pts.remove()
        self.adjust_variables()
        self.adjust_plots()



    def adjust_variables(self):
        self.img[:] = 0
        self.idx["roi"] = []


    def adjust_plots(self):
        self.pts = self.ax1.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c="blue")
        
        self.ax2.cla()
        self.ax2.axis("off")
        self.ax2.set_title('Pixel Selection:')
        self.ax2.imshow(self.img, vmax=1)

    
    def return_rois(self, event=None):
        plt.close()
        return self.rois