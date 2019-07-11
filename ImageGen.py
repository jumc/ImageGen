

import math
import random
import cairo
from colorharmonies import Color, complementaryColor, triadicColor, splitComplementaryColor, tetradicColor, analogousColor, monochromaticColor

class ImageGen:
    def set_random_shape(self, ctx):
        shape_type = random.randrange(1,5) # type 1: line, type 2: rectangle, type 3: arc/circle/elipse, type 4: complex shape,
        if shape_type is 1:
            ctx.move_to(random.random(), random.random()) # move_to(x, y)
            ctx.line_to(random.random(), random.random()) # line_to(x, y)
            ctx.move_to(0, 0)
        if shape_type is 2:
            ctx.rectangle(random.random(), random.random(), random.random(), random.random())  # Rectangle(x0, y0, x1, y1)
        if shape_type is 3:
            x, y, width, height = random.random(), random.random(), random.random(), random.random()
            ctx.save()
            ctx.translate(x + width / 2., y + height / 2.)
            ctx.scale(width / 2., height / 2.)
            ctx.arc(0., 0., 1., random.random() * 2 * math.pi, random.random() * 2 * math.pi) # arc(xc, yc, radius, angle1, angle2)
            ctx.restore()
        if shape_type is 4:
            n_steps = random.randrange(1, 50)
            ctx.move_to(random.random(), random.random()) # move_to(x, y)
            for i in range(n_steps):
                ctx.rel_curve_to(random.random(), random.random(), random.random(), random.random(),random.random(), random.random()) # rel_curve_to(x1, y1, x2, y2, x3, y3)
            ctx.close_path()
            ctx.move_to(0, 0)

    def randomColor(self, main, type):
      perturbation = [random.randrange(-50,50), random.randrange(-50,50), random.randrange(-50,50)]
      if type is 0: # random color_scheme
          type = random.randrange(1, 8)
      if type is 1:
          color = complementaryColor(main)
      if type is 2:
          color = triadicColor(main)[random.randrange(0,2)]
      if type is 3:
          color = splitComplementaryColor(main)[random.randrange(0,2)]
      if type is 4:
          color = tetradicColor(main)[random.randrange(0,3)]
      if type is 5:
          color = analogousColor(main)[random.randrange(0,2)]
      if type is 6:
          color = monochromaticColor(main)[random.randrange(0,10)]
      if type is 7: # true random
          return [random.random(), random.random(), random.random()]
      return [(color[0] + perturbation[0])/255, (color[1] + perturbation[1])/255, (color[2] + perturbation[2])/255]

    def randomColorShape(self, ctx, bg_color, color_scheme):
        draw_type = random.randrange(1,4) # type 1: fill, type 2: stroke, type 3: stroke and fill
        color = self.randomColor(bg_color, color_scheme)
        ctx.set_source_rgb(color[0], color[1], color[2])  # Solid color
        if draw_type is 1:
            ctx.fill()
        elif draw_type is 3:
            ctx.fill_preserve()
        if draw_type is not 1:
            color = self.randomColor(bg_color, color_scheme)
            ctx.set_line_width(random.random()/((random.random()+0.01)*100))
            ctx.set_source_rgb(color[0], color[1], color[2])  # Solid color
            ctx.stroke()

    def drawImg(self, width, height, min_shapes, max_shapes, path):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
        ctx = cairo.Context(surface)
        ctx.scale(width, height)  # Normalizing the canvas

        bg_color = Color([random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)], "", "")
        ctx.set_source_rgb(bg_color.RGB[0]/255, bg_color.RGB[1]/255, bg_color.RGB[2]/255)  # Background
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        n_shapes = random.randrange(min_shapes, max_shapes + 1)
        color_scheme = random.randrange(0, 8)

        for shape in range(n_shapes):
            self.set_random_shape(ctx)
            self.randomColorShape(ctx, bg_color, color_scheme)

        surface.write_to_png(path)  # Output to PNG


PATH = "generatedImgs/"
n_images = 10000
imageGen = ImageGen()
for i in range(n_images):
    imageGen.drawImg(256, 256, 5, 40, PATH + str(i) + ".png")
