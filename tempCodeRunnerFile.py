import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from io import BytesIO

# Extract color palette with guaranteed dark/light color detection
def extract_color_palette(image, k=6):
    image = image.resize((200, 200))
    data = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_.astype(int)
    
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    
    # Sort colors by frequency
    color_freq = [(centers[i], label_counts[i]/total) for i in label_counts]
    color_freq.sort(key=lambda x: -x[1])
    colors = [c for c, _ in color_freq]

    # Ensure black and white are preserved if present
    def is_close(color1, color2, threshold=30):
        return np.linalg.norm(np.array(color1) - np.array(color2)) < threshold

    guaranteed_colors = []
    for special in [(0, 0, 0), (255, 255, 255)]:
        for pixel in data:
            if is_close(pixel, special, 30):
                guaranteed_colors.append(special)
                break

    for gc in guaranteed_colors:
        if all(not is_close(gc, c) for c in colors):
            colors.append(gc)

    return colors[:6]

# Show the palette
def show_palette(colors):
    fig, ax = plt.subplots(figsize=(6, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Handle image selection
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img.thumbnail((400, 400))

    img_display = ImageTk.PhotoImage(img)
    image_label.config(image=img_display)
    image_label.image = img_display

    colors = extract_color_palette(img)
    palette_img = show_palette(colors)
    palette_display = ImageTk.PhotoImage(palette_img)
    palette_label.config(image=palette_display)
    palette_label.image = palette_display

# GUI setup
app = tk.Tk()
app.title("Color Palette Extractor")
app.geometry("1000x600")
app.configure(bg="#2d2d2d")

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", foreground="black", background="#dddddd", font=("Arial", 12))

open_btn = ttk.Button(app, text="Open Image", command=open_image)
open_btn.pack(pady=10)

frame = tk.Frame(app, bg="#2d2d2d")
frame.pack(fill="both", expand=True)

left_frame = tk.LabelFrame(frame, text="Selected Image", bg="#ccc", font=("Arial", 10))
left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

right_frame = tk.LabelFrame(frame, text="Extracted Color Palette", bg="white", font=("Arial", 10))
right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

image_label = tk.Label(left_frame, bg="#ccc")
image_label.pack(expand=True)

palette_label = tk.Label(right_frame, bg="white")
palette_label.pack(expand=True)

app.mainloop()
