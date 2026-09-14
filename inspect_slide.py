import openslide

slide = openslide.OpenSlide("camelyon16_subset/normal/normal_001.tif")

print("=== All slide properties ===")
for key, value in slide.properties.items():
    print(f"{key}: {value}")

print("\n=== Level info ===")
print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)
print("Level downsamples:", slide.level_downsamples)

slide.close()
