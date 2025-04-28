from PIL import Image

def draw_point(image, x, y, color):
    if 0 <= x < image.width and 0 <= y < image.height:
        image.putpixel((x, y), color)

def draw_line(image, x1, y1, x2, y2, color):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    xi = 1 if x2 - x1 > 0 else -1 if x2 - x1 < 0 else 0
    yi = 1 if y2 - y1 > 0 else -1 if y2 - y1 < 0 else 0

    x = x1
    y = y1

    draw_point(image, x, y, color)  # Rysujemy punkt początkowy

    d = 2 * dy - dx
    if dx > dy:
        while x != x2:
            x += xi
            if d >= 0:
                y += yi
                d -= 2 * dx
            d += 2 * dy
            draw_point(image, x, y, color)
    else:
        while y != y2:
            y += yi
            if d >= 0:
                x += xi
                d -= 2 * dy
            d += 2 * dx
            draw_point(image, x, y, color)


def area(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def is_inside_triangle(p, v0, v1, v2):
    a = area(v0, v1, p)
    b = area(v1, v2, p)
    c = area(v2, v0, p)
    return (a >= 0 and b >= 0 and c >= 0) or (a <= 0 and b <= 0 and c <= 0)

def draw_filled_triangle(image, v0, v1, v2, color):
    xmin = int(min(v0[0], v1[0], v2[0]))
    xmax = int(max(v0[0], v1[0], v2[0]))
    ymin = int(min(v0[1], v1[1], v2[1]))
    ymax = int(max(v0[1], v1[1], v2[1]))

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if is_inside_triangle((x, y), v0, v1, v2):
                draw_point(image, x, y, color)


img = Image.new("RGB", (200, 200), (0, 0, 0))

# Rysuj linię
draw_line(img, 10, 10, 190, 150, (255, 0, 0))

# Rysuj trójkąt
v0 = (50, 50)
v1 = (150, 30)
v2 = (100, 170)
draw_filled_triangle(img, v0, v1, v2, (0, 0, 255))

img.save("line_triangle.png")
img.show()
