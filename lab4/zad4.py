from PIL import Image

def draw_point(image, x, y, color):
    width, height = image.size
    if 0 <= x < width and 0 <= y < height:
        image.putpixel((x, y), color)

def interpolate_color(color1, color2, t):
    r = round((1 - t) * color1[0] + t * color2[0])
    g = round((1 - t) * color1[1] + t * color2[1])
    b = round((1 - t) * color1[2] + t * color2[2])
    return (r, g, b)

def draw_line_interpolated(image, x0, y0, color0, x1, y1, color1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    xi = 1 if x1 - x0 > 0 else -1 if x1 - x0 < 0 else 0
    yi = 1 if y1 - y0 > 0 else -1 if y1 - y0 < 0 else 0

    x = x0
    y = y0

    draw_point(image, x, y, color0)

    if dx > dy:
        d = 2 * dy - dx
        steps = dx
        for i in range(steps):
            x += xi
            if d >= 0:
                y += yi
                d -= 2 * dx
            d += 2 * dy
            t = i / steps
            color = interpolate_color(color0, color1, t)
            draw_point(image, x, y, color)
    else:
        d = 2 * dx - dy
        steps = dy
        for i in range(steps):
            y += yi
            if d >= 0:
                x += xi
                d -= 2 * dy
            d += 2 * dx
            t = i / steps
            color = interpolate_color(color0, color1, t)
            draw_point(image, x, y, color)

def area(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def draw_filled_triangle_interpolated(image, v0, color0, v1, color1, v2, color2):
    xmin = int(min(v0[0], v1[0], v2[0]))
    xmax = int(max(v0[0], v1[0], v2[0]))
    ymin = int(min(v0[1], v1[1], v2[1]))
    ymax = int(max(v0[1], v1[1], v2[1]))

    total_area = area(v0, v1, v2)

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            p = (x, y)
            w0 = area(p, v1, v2) / total_area
            w1 = area(p, v2, v0) / total_area
            w2 = area(p, v0, v1) / total_area

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                r = round(w0 * color0[0] + w1 * color1[0] + w2 * color2[0])
                g = round(w0 * color0[1] + w1 * color1[1] + w2 * color2[1])
                b = round(w0 * color0[2] + w1 * color1[2] + w2 * color2[2])
                draw_point(image, x, y, (r, g, b))


width, height = 300, 300
image = Image.new("RGB", (width, height), (255, 255, 255))

draw_line_interpolated(image, 50, 50, (255, 0, 0), 250, 100, (0, 0, 255))

v0 = (100, 150)
v1 = (200, 180)
v2 = (150, 250)
color0 = (255, 0, 0)
color1 = (0, 255, 255)
color2 = (255, 0, 255)

draw_filled_triangle_interpolated(image, v0, color0, v1, color1, v2, color2)

image.save("output.png")
