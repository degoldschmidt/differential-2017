import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tkinter import filedialog

from os.path import expanduser
HOME = expanduser("~")

class GUI(object):
    def __init__(self, master, res):
        # Create a canvas
        self.guiw, self.guih = res
        print(self.guiw, self.guih)
        self.master = master
        self.master.minsize(self.guiw, self.guih)
        self.master.title("PyTrack-preview")
        menubar = self.init_menu()

        # app data (trajectories and filelist)
        self.cols = ['Item1.Item1.X', 'Item1.Item1.Y']
        self.X = []
        self.Y = []
        self.filelist = []

        # display the menu
        self.master.config(menu=menubar)
        self.canvas, self.f, self.ax = self.init_canvas()
        self.listbox = self.init_filelist()
        self.listbox.bind('<Double-Button-1>', self.get_filename)

    def about(self):
        pass

    def get_filename(self, event):
        """
        function to read the listbox selection
        and put the result in an entry widget
        """
        if len(self.listbox.curselection()) == 1:
            # get selected line index
            index = self.listbox.curselection()[0]
            # get the line's text
            seltext = self.listbox.get(index)
            # load data and plot
            if seltext.endswith('csv'):
                data = pd.read_csv(seltext, sep=' ', index_col=False)
                self.X, self.Y = data.loc[:, self.cols[0]], data.loc[:, self.cols[1]]
                self.update_ax()



    def init_canvas(self):
        canvas = tk.Canvas(self.master, width=self.guih, height=self.guih)
        canvas.pack(side=tk.LEFT)
        # Create the figure we desire to add to an existing canvas
        f, ax = plt.subplots(figsize=(self.guih/100, self.guih/100))

        # Keep this handle alive, or else figure will disappear
        f_x, f_y = 100, 0
        fig_photo = draw_figure(canvas, f, loc=(f_x, f_y))
        f_w, f_h = fig_photo.width(), fig_photo.height()
        return canvas, f, ax

    def init_filelist(self):
        listbox = tk.Listbox(self.master)
        listbox.pack(fill=tk.BOTH, expand=1)
        for item in self.filelist:
            listbox.insert(tk.END, item)
        return listbox

    def init_menu(self):
        menubar = tk.Menu(self.master)
        #### MENUBAR
        # create a pulldown menu, and add it to the menu bar
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load", command=self.load)
        #filemenu.add_command(label="Save", command=self.)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        return menubar

    def load(self):
        self.filelist = filedialog.askopenfilenames(initialdir="/Users/degoldschmidt/Google Drive/PhD Project/Experiments/001-DifferentialDeprivation/data/", title="Select files")
        self.update_filelist()
        if len(self.cols) != 2:
            self.select_data()

    def put_data(self):
        Nsel = sum([each.get() for each in self.colvars])
        if Nsel == 2:
            indices = np.where([each.get() for each in self.colvars])[0]
            self.cols = [self.buffer.columns[i] for i in indices]
            print(self.cols)
            self.selectwindow.destroy()

    def select_data(self):
        self.selectwindow = tk.Toplevel(self.master)
        self.selectwindow.wm_title("New Window")

        if self.filelist[0].endswith('csv'):
            self.buffer = pd.read_csv(self.filelist[0], sep=' ', index_col=False)
            datacols = self.buffer.columns
            tk.Label(self.selectwindow, text="Select data columns (only 2):").grid(row=0, sticky=tk.W)
            self.colvars = []
            for i,each in enumerate(datacols):
                self.colvars.append(tk.BooleanVar())
                tk.Checkbutton(self.selectwindow, text=each, variable=self.colvars[-1]).grid(row=i+1, sticky=tk.W)
        else:
            tk.Label(self.selectwindow, text="Select valid csv files!").grid(row=0, sticky=tk.W)
        tk.Button(self.selectwindow, text='Select', command=self.put_data).grid(row=len(datacols)+1, sticky=tk.W, pady=4)
        #Button(master, text='Show', command=var_states).grid(row=4, sticky=tk.W, pady=4)

    def update_ax(self):
        self.ax.cla()
        x, y = self.X, self.Y
        self.ax.plot(x, y)
        ### checking data limits
        xxrange = np.nanmax(x)-np.nanmin(x)
        yyrange = np.nanmax(y)-np.nanmin(y)
        rrange = np.max([xxrange, yyrange])
        xc = (np.nanmax(x)+np.nanmin(x))/2
        yc = (np.nanmax(y)+np.nanmin(y))/2
        self.ax.set_xlim([xc-1.05*rrange/2,xc+1.05*rrange/2])
        self.ax.set_ylim([yc-1.05*rrange/2,yc+1.05*rrange/2])
        f_x, f_y = 0, 0
        self.fig_photo = draw_figure(self.canvas, self.f, loc=(f_x, f_y))

    def update_filelist(self):
        for item in self.filelist:
            self.listbox.insert(tk.END, item)


def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo

# Let Tk take over
root = tk.Tk()
w,h = root.winfo_screenwidth(), root.winfo_screenheight()
_ = GUI(root, (w,h-200))
root.mainloop()
