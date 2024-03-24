import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from predict import predict_image, CarBikeClassifier

app = tk.Tk()
app.geometry("700x700")
app.configure(bg="lightgray")

def select_file():
    filename = filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file = filename
    img = Image.open(filename)
    img.thumbnail((300, 400))  # Resize the image to fit in the UI
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img

def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text = predict_image(model, image_file, device='cpu')

    result_label.configure(text=prediction_text, fg="white", bg="green", font=("Helvetica", 14, "bold"))

image_label = tk.Label(app, bg="white")
image_label.place(relx=0.5, rely=0.3, anchor="center")

choose_file_button = tk.Button(app, text="Choose file", command=select_file, bg="black", fg="white", font=("Helvetica", 12, "bold"))
choose_file_button.place(relx=0.5, rely=0.9, anchor="center")

classify_button = tk.Button(app, text="Classify", command=classify, bg="orange", fg="white", font=("Helvetica", 12, "bold"))
classify_button.place(relx=0.7, rely=0.7, anchor="center")

result_label = tk.Label(app, text="", bg="lightblue", font=("Helvetica", 12))
result_label.place(relx=0.8, rely=0.1, anchor="center")

app.mainloop()
