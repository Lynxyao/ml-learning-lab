from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


PDF_PATH = Path(
    r"C:\Users\10131\Documents\URAP 2\resistance_results\reports\hybrid_vs_mna_gnn"
    r"\Module4_Hybrid_GNN_vs_MNA_GNN_Group_Meeting_Report.pdf"
)
OUTPUT_DIR = PDF_PATH.parent / "page_previews"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(str(PDF_PATH))
    thumbnails = []
    for index, page in enumerate(document):
        image = page.render(scale=1.2).to_pil().convert("RGB")
        image.save(OUTPUT_DIR / f"page_{index + 1:02d}.png")
        image.thumbnail((520, 680))
        thumbnails.append(image.copy())
    for start in range(0, len(thumbnails), 4):
        sheet = Image.new("RGB", (1080, 1400), "#dfe4e8")
        draw = ImageDraw.Draw(sheet)
        for index in range(start, min(start + 4, len(thumbnails))):
            position = index - start
            x = (position % 2) * 540 + 10
            y = (position // 2) * 700 + 18
            sheet.paste(thumbnails[index], (x, y))
            draw.text((x + 8, y - 16), f"Page {index + 1}", fill="#17324d")
        sheet.save(OUTPUT_DIR / f"contact_{start // 4 + 1:02d}.png")
    print(f"Rendered {len(thumbnails)} pages to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
