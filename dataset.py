from sklearn.datasets import load_digits

ALPH = "0123456789ABCDEFG"  # 17 chars for values 0..16

digits = load_digits()  # 8x8, values 0..16 [web:195]
imgs = digits.images

with open("ascii_dataset.txt", "w", encoding="utf-8") as f:
    for img in imgs:
        for y in range(8):
            row = "".join(ALPH[int(img[y, x])] for x in range(8))
            f.write(row + "\n")
        f.write("\n")  # blank line -> doc separator (so split by \n\n works)
print("Saved ascii_dataset.txt")
