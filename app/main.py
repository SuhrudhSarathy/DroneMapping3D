from logging import log
from tkinter import *
from tkinter import filedialog as fd
from backend import *
from backend import classifier
from PIL import ImageTk, Image
import datetime
import os

BUTTON_H = 8
BUTTON_W = 10

class App:
    def __init__(self, master: Tk) -> None:
        self.master = master
        self.master.title('Species Identification Model')
        self.master.geometry("1080x700")

        # set minimum window size value
        self.master.minsize(1080, 700)
        
        # # set maximum window size value
        self.master.maxsize(1080, 700)

        # Button Frame lies on the bottom of the window
        self.buttonFrame = Frame(self.master, width=100)
        self.buttonFrame.pack(side=RIGHT, fill="both")

        self.mainFrame = PanedWindow(self.master)
        self.mainFrame.pack(side=LEFT, fill="both", expand=True)

        # Buttons

        self.select_file_button = Button(self.buttonFrame, text="Select File", command=self.select_file, height=BUTTON_H, width=BUTTON_W)
        self.select_file_button.grid(row=0, column=0)

        self.select_folder_button = Button(self.buttonFrame, text="Select Folder", command=self.select_folder, height=BUTTON_H, width=BUTTON_W)
        self.select_folder_button.grid(row=1, column=0)

        self.select_destination = Button(self.buttonFrame, text="Select Destination", command=self.select_destination, height=BUTTON_H, width=BUTTON_W)
        self.select_destination.grid(row=2, column=0)

        self.predict_button = Button(self.buttonFrame, text="Predict", command=self.predict, height=BUTTON_H, width=BUTTON_W)
        self.predict_button.grid(row=3, column=0)

        self.clear_button = Button(self.buttonFrame, text="Clear", command=self.clear_selection, height=BUTTON_H, width=BUTTON_W)
        self.clear_button.grid(row=4, column=0)

        # Canvas
        self.canvas = Canvas(self.mainFrame, height=678, width=980)
        self.canvas.pack()
        self.bits_logo = PhotoImage(file="BITS_Goa_campus_logo.gif")

        self.canvas.create_image((350, 250), image=self.bits_logo, anchor=NW)

        # set the buttons as disabled
        self.clear_button["state"] = DISABLED
        self.predict_button["state"] = DISABLED

        self.file_name = ""
        self.folder_name = ""
        self.destination_name = ""

        self.file_types = (
            ('jpgs', '*.jpg'),
            ('pngs', '*.png'),
            ('JPEGS', '*.JPEG')
        )

        # Variables to keep track of what is selected
        self.file_selected = False
        self.folder_selected = False
        self.destination_selected = False

    def select_file(self):
        self.file_name = fd.askopenfilename(
            title="Select File",
            filetypes=self.file_types
        )
        self.file_name = self.file_name.strip()

        if len(self.file_name) <= 3:
            self.file_selected = False
        else:
            self.file_selected = True
            self.canvas.delete("all")

            self.result_img = Image.open(self.file_name)
            self.result_img = self.result_img.resize((800, 600))
            self.result_img = ImageTk.PhotoImage(self.result_img)
            self.canvas.create_image(60, 50, image=self.result_img, anchor=NW)


        if self.destination_selected:
            self.predict_button["state"] = NORMAL
            self.clear_button["state"] = NORMAL
        else:
            self.predict_button["state"] = DISABLED
            self.clear_button["state"] = DISABLED

        self.select_folder_button["state"] = DISABLED

    def select_folder(self):
        self.folder_name = fd.askdirectory()

        if self.destination_selected:
            self.predict_button["state"] = NORMAL
            self.clear_button["state"] = NORMAL
        else:
            self.predict_button["state"] = DISABLED
            self.clear_button["state"] = DISABLED

        self.folder_selected = True

        self.select_file_button["state"] = DISABLED

    def select_destination(self):
        self.destination_name = fd.askdirectory()

        print(self.destination_name)

        if self.file_selected or self.folder_selected:
            self.predict_button["state"] = NORMAL
            self.clear_button["state"] = NORMAL
        else:
            self.predict_button["state"] = DISABLED
            self.clear_button["state"] = DISABLED

        self.destination_selected = True


    def predict(self):
        self.predict_button["state"] = DISABLED
        self.clear_button['state'] = NORMAL

        try:

            # Run the nueral network based on what is selected
            if self.file_selected:
                time = datetime.datetime.now()
                now = time.strftime("%d-%m-%Y::%H:%M:%S")
                fi_path = os.path.join(self.destination_name, f"{now}.png")
                f_path = os.path.join(self.destination_name, f"{now}.log")

                decorate_results(self.file_name,fi_path, f_path)

                # Delete whatever is there in the canvas and display the results instead
                self.canvas.delete("all")

                self.result_img = Image.open(fi_path)
                self.result_img = self.result_img.resize((800, 600))
                self.result_img = ImageTk.PhotoImage(self.result_img)
                self.canvas.create_image(50, 50, image=self.result_img, anchor=NW)

            elif self.folder_selected:
                log_file = os.path.join(self.destination_name, "results.log")
                classifier.predict_from_folder(self.folder_name, log_file)

        except Exception as e:
            print(e)
            print("Bye Bye")
            self.master.destroy()

        # Once prediction is over, call the clear function
        self.select_file_button["state"] = NORMAL
        self.select_folder_button["state"] = NORMAL
    
    def clear_selection(self, canvas=None):
        self.predict_button["state"] = DISABLED
        self.clear_button['state'] = NORMAL
        self.select_file_button["state"] = NORMAL
        self.select_folder_button["state"] = NORMAL

        self.folder_selected = False
        self.file_selected = False
        self.file_name = ""
        self.folder_name = ""

        self.canvas.delete("all")
        self.canvas.create_image((350, 250), image=self.bits_logo, anchor=NW)


if __name__ == "__main__":
    root = Tk()
    app = App(root)

    root.mainloop()
