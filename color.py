import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from collections import Counter
from io import BytesIO
import colorsys
from scipy.spatial import KDTree

class ColorPaletteExtractor:
    def __init__(self):
        self.COLOR_THRESHOLD = 45
        self.MIN_SATURATION = 0.15
        self.MIN_VALUE = 0.15
        self.PALETTE_SIZE = 8

    def extract_dominant_colors(self, image, k=16):
        """استخراج رنگ‌های غالب با K-Means++"""
        image = image.resize((400, 400))
        pixels = np.array(image).reshape(-1, 3)
        
        sample_size = 2000
        if len(pixels) > sample_size:
            pixels = shuffle(pixels, random_state=42)[:sample_size]
        
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=20,
            max_iter=300,
            tol=1e-4,
            random_state=42
        )
        kmeans.fit(pixels)
        
        counts = Counter(kmeans.labels_)
        total = sum(counts.values())
        return [
            (kmeans.cluster_centers_[i].astype(int), counts[i]/total)
            for i in range(k)
        ]

    def get_hsv_ranges(self):
        """محدوده‌های HSV برای رنگ‌های کلیدی"""
        return {
            'red': [
                {'lower': (0, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.05, 1.0, 1.0)},
                {'lower': (0.9, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (1.0, 1.0, 1.0)}
            ],
            'green': [{'lower': (0.25, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.45, 1.0, 1.0)}],
            'blue': [{'lower': (0.55, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.75, 1.0, 1.0)}],
            'yellow': [{'lower': (0.10, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.20, 1.0, 1.0)}],
            'purple': [{'lower': (0.75, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.85, 1.0, 1.0)}],
            'orange': [{'lower': (0.05, self.MIN_SATURATION, self.MIN_VALUE), 'upper': (0.10, 1.0, 1.0)}],
            'black': [{'lower': (0, 0, 0), 'upper': (1.0, 1.0, 0.20)}],
            'white': [{'lower': (0, 0, 0.85), 'upper': (1.0, 0.25, 1.0)}]
        }

    def detect_special_colors(self, pixels):
        """تشخیص رنگ‌های خاص"""
        hsv_pixels = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in pixels/255.0])
        detected = set()
        
        for color_name, ranges in self.get_hsv_ranges().items():
            for r in ranges:
                lower = np.array(r['lower'])
                upper = np.array(r['upper'])
                
                mask = np.ones(len(hsv_pixels), dtype=bool)
                for i in range(3):
                    if lower[i] <= upper[i]:
                        mask &= (hsv_pixels[:,i] >= lower[i]) & (hsv_pixels[:,i] <= upper[i])
                    else:
                        mask &= (hsv_pixels[:,i] >= lower[i]) | (hsv_pixels[:,i] <= upper[i])
                
                if np.any(mask):
                    detected.add(color_name)
                    break
        
        return detected

    def merge_similar_colors(self, colors):
        """ادغام رنگ‌های مشابه"""
        if colors.size == 0:
            return []

        kdtree = KDTree(colors)
        clusters = []
        used = set()

        for i, color in enumerate(colors):
            if i not in used:
                neighbors = kdtree.query_ball_point(color, r=self.COLOR_THRESHOLD)
                cluster = [colors[j] for j in neighbors]
                clusters.append(np.mean(cluster, axis=0).astype(int))
                used.update(neighbors)

        return clusters

    def generate_palette(self, image):
        """تولید پالت نهایی"""
        dominant_colors = self.extract_dominant_colors(image)
        dominant_colors.sort(key=lambda x: -x[1])
        raw_colors = np.array([c[0] for c in dominant_colors])

        pixels = np.array(image.resize((100, 100))).reshape(-1, 3)
        special_colors = self.detect_special_colors(pixels)
        
        color_map = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        
        special_rgb = [color_map[c] for c in special_colors if c in color_map]
        
        if special_rgb:
            all_colors = np.vstack([raw_colors, special_rgb])
        else:
            all_colors = raw_colors
            
        merged_colors = self.merge_similar_colors(all_colors)
        
        final_palette = []
        for color in merged_colors:
            if len(final_palette) >= self.PALETTE_SIZE:
                break
            if not any(np.linalg.norm(color - c) < self.COLOR_THRESHOLD for c in final_palette):
                final_palette.append(tuple(color))
        
        return final_palette[:self.PALETTE_SIZE]

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.extractor = ColorPaletteExtractor()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Professional Color Palette Extractor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#333")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 12), padding=6)
        style.map("TButton", background=[("active", "#666"), ("!disabled", "#444")])

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.open_btn = ttk.Button(control_frame, text="Open Image", command=self.load_image)
        self.open_btn.pack(side=tk.LEFT)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.image_panel = ttk.Label(display_frame, background="#444")
        self.image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.palette_panel = ttk.Label(display_frame, background="#fff")
        self.palette_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")
        ])
        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
            self.display_image(image)
            palette = self.extractor.generate_palette(image)
            self.display_palette(palette)
        except Exception as e:
            print(f"Error processing image: {e}")

    def display_image(self, image):
        image.thumbnail((600, 600))
        img_tk = ImageTk.PhotoImage(image)
        self.image_panel.config(image=img_tk)
        self.image_panel.image = img_tk

    def display_palette(self, colors):
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_title("Color Palette", pad=20, fontsize=14)
        
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255))
            hex_code = '#%02x%02x%02x' % color
            ax.text(i + 0.5, -0.3, hex_code, ha='center', va='top', fontsize=10)
        
        ax.set_xlim(0, len(colors))
        ax.set_ylim(-0.5, 1)
        ax.axis("off")
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        palette_img = Image.open(buf)
        palette_tk = ImageTk.PhotoImage(palette_img)
        self.palette_panel.config(image=palette_tk)
        self.palette_panel.image = palette_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()

