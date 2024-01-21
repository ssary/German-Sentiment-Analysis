from pathlib import Path
from tkinter import Tk, END, Canvas, Entry, Text, Button, PhotoImage, StringVar
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()
window.geometry("700x550")
window.configure(bg = "#D9D9D9")

sentiment_text_area = StringVar("")
review_sentiment = StringVar(value="Sentiment")

canvas = Canvas(
    window,
    bg = "#D9D9D9",
    height = 550,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    350.0,
    32.0,
    image=image_image_1
)

canvas.create_text(
    129.0,
    17.0,
    anchor="nw",
    text="Pankow Reviews Sentiment Analysis",
    fill="#C8C8C8",
    font=("Itim Regular", 27 * -1)
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    351.0,
    233.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=29.0,
    y=143.0,
    width=644.0,
    height=178.0
)

def get_entry_data():
    return entry_1.get("1.0", END)

canvas.create_text(
    24.0,
    89.0,
    anchor="nw",
    text="Enter your review",
    fill="#414141",
    font=("Itim Regular", 28 * -1)
)

canvas.create_text(
    23.0,
    422.0,
    anchor="nw",
    text="Sentiment",
    fill="#414141",
    font=("Itim Regular", 31 * -1)
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    350.0,
    501.0,
    image=image_image_2
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))

def take_input():
    from controller import handle_on_click
    new_sentiment = handle_on_click(entry_1.get("1.0", END))
    review_sentiment.set(value=new_sentiment)
    canvas.itemconfig(sentiment_text_label, text=review_sentiment.get())

button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=take_input,
    relief="flat"
)
button_1.place(
    x=212.0,
    y=358.0,
    width=276.0,
    height=48.0
)

sentiment_text_label = canvas.create_text(
    280.0,
    484.0,
    anchor="nw",
    text=review_sentiment.get(),
    fill="#D9D9D9",
    font=("Itim Regular", 31 * -1)
)
window.resizable(False, False)
def main_loop():
    window.mainloop()
