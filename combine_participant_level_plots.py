
from pathlib import Path
import math
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cairosvg

import re

def extract_last_int(path: Path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else float("inf")

def combine_svgs(
    svg_dir,
    output_path="combined.png",
    dpi=300,
    padding=20,
    cols=1,
    font_size=40,
    background=(255, 255, 255)
):
    title_height = int(1.6 * font_size)
    svg_files = svg_dir.glob("*.svg")
    svg_files = sorted(list(svg_files), key=extract_last_int)  # Sort by participant number

    if not svg_files:
        raise RuntimeError("No SVG files found.")

    images = []

    # Convert SVGs to PIL Images
    for svg in svg_files:
        svg_path = str(svg)
        png_bytes = cairosvg.svg2png(url=svg_path, dpi=dpi)
        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
        images.append(img)

    n = len(images)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    cell_width = max(img.width for img in images)
    image_height = max(img.height for img in images)
    cell_height = title_height + image_height

    total_width = cols * cell_width + padding * (cols - 1)
    total_height = rows * cell_height + padding * (rows - 1)

    combined = Image.new("RGBA", (total_width, total_height), background)
    draw = ImageDraw.Draw(combined)

    # Try to load a real font; fall back cleanly
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, (img, fname) in enumerate(zip(images, svg_files)):
        r = idx // cols
        c = idx % cols

        cell_x = c * (cell_width + padding)
        cell_y = r * (cell_height + padding)

        # Title
        title = f"Participant {extract_last_int(fname)}"  # Get participant name from filename

        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        text_x = cell_x + (cell_width - text_width) // 2
        text_y = cell_y + (title_height - font_size) // 2

        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

        # Image (centered under title)
        img_x = cell_x + (cell_width - img.width) // 2
        img_y = cell_y + title_height + (image_height - img.height) // 2

        combined.paste(img, (img_x, img_y), img)

    combined.convert("RGB").save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    path = Path(__file__).parent / "results"

    for results_dir, cols in zip(["h1", "h2", "posthoc/h1_response", "posthoc/h2_response"], [3, 5, 3, 5]):
        print(f"Processing {results_dir}...")
        # split on / if there is one, and take the last part for the output filename
        output_file = path / results_dir / f"participant_level_combined_{results_dir.split('/')[-1]}.png"
        combine_svgs( path / results_dir / "participant_level", output_path=output_file, cols=cols)
            
            
