import re
import base64
from pathlib import Path

html_file = Path("analysis-scripts/analysis-script.html")
html_content = html_file.read_text(encoding='utf-8')

# Find all base64-encoded images
pattern = r'src="data:image/png;base64,([^"]*)"'
matches = re.findall(pattern, html_content)

print(f"Found {len(matches)} images")

# Save each image
for i, base64_data in enumerate(matches, 1):
    try:
        image_data = base64.b64decode(base64_data)
        
        # Determine filename based on position
        if i == 1:
            filename = "plot_nonce_binomials.png"
        elif i == 2:
            filename = "plot_attested_binomials.png"
        elif i == 3:
            filename = "plot_frequency_analysis.png"
        else:
            filename = f"plot_{i}.png"
        
        output_path = Path(filename)
        output_path.write_bytes(image_data)
        print(f"Saved {filename} ({len(image_data)} bytes)")
    except Exception as e:
        print(f"Error saving image {i}: {e}")

print("Done!")
