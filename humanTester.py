import tkinter as tk
from PIL import Image, ImageTk
import random

padding = 40
picwidth = 224

class App():

    def random_image(self):
        self.current_label = random.randint(0, 1)
        self.photo = ImageTk.PhotoImage(Image.open('./train_data_%s/%s.jpg' % (self.current_label, str(random.randint(1, 10000)))))

    def show_img(self):
        self.image_on_canvas = self.canvas.create_image(padding + picwidth / 2 , padding , anchor='n',image=self.photo)

    def keyevent(self, event):
        if event.keycode == 37:
            self.click(1)
        else:
            self.click(0)

    def click(self, x):
        print('click', x, x == self.current_label)
        self.counter[self.current_label][x] += 1
        self.sum += 1

        print('acc={:4f}'.format((self.counter[1][1] + self.counter[0][0]) / self.sum), end="\t")
        for i in range(2):
            for j in range(2):
                print("({}->{})".format(i, j), self.counter[i][j], end="\t")
        print()
        self.canvas.delete(self.image_on_canvas)
        self.random_image()
        
        self.show_img()

    def __init__(self):
        self.counter = [[0, 0], [0, 0]]
        self.sum = 0

        self.root = tk.Tk()
        self.root.title("humanTester")
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        width = padding * 2 + picwidth
        height =  picwidth * 2
        alignstr = "%dx%d+%d+%d" % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.root.geometry(alignstr)
        self.root.resizable(width=True, height=True)

        self.canvas = tk.Canvas(self.root, height=width, width=width, bg="black")
        self.random_image()
        
        self.show_img()
        self.canvas.grid(row=0,column=0,columnspan=2)

        self.b1 = tk.Button(self.root, text='MIX', font=("Consolas", 28), fg="white", bg="red", width=7, height=2, command=lambda : self.click(1))
        self.b2 = tk.Button(self.root, text='RAW', font=("Consolas", 28), fg="white", bg="green", width=7,  height=2, command=lambda : self.click(0))
        self.b1.grid(row=1,column=0)
        self.b2.grid(row=1,column=1)
        
        self.root.bind("<Left>", self.keyevent)
        self.root.bind("<Right>", self.keyevent)
        

if __name__ == "__main__":
    app = App()
    app.root.mainloop()
