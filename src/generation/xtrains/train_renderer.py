import math

import numpy as np
from skimage import draw, transform
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte

"""
Class responsible for generating a train image based on a train configuration.
Train configurations are generated in ``config_generator.py''. It also adds information to the train configuration
about where each part of the train is drawn in the generated image.
"""

LOCOMOTIVE_WIDTH = 13
LOCOMOTIVE_HEIGHT = 20
IMG_WID = 152
IMG_HEI = 152
THICK_PROB = 0.15  # percentage of thick walls
MAX_NOISE = 0.1  # percentage of max noisy pixels
PADDING = 20  # number of pixels to account for the rotation
BACKGROUND_TRIES = 100  # number of times we try to find a background that is not almost completely black
                        # (prevents from entering an infinite loop if only black images are provided as background)


class TrainRenderer:
    def __init__(self, train_config):
        self.train_config = train_config

        self.seed = train_config['render_seed'] if train_config['render_seed'] is not None else 42
        self.rng_n = np.random.default_rng(self.seed + 1) # seed stream for noise
        self.rng_b = np.random.default_rng(self.seed + 2) # seed stream for background
        self.rng_p = np.random.default_rng(self.seed + 3) # seed stream for position


        # each 'position box' represented as (top_left, top_right, bottom_right, bottom_right)
        self.train_config['train_pos'] = []
        self.train_config['cars_pos'] = []
        self.train_config['cargos_pos'] = []
        self.train_config['roofs_pos'] = []
        self.train_config['wheels_pos'] = []
        self.train_config['cargo_belonging'] = []

        self.car_imgs = []
        self.train_img = np.ones((IMG_HEI, IMG_WID, 3))

        if 'background' in self.train_config:
            self.background = imread(self.train_config['background'])
        else:
            self.background = None

        # print(self.train_config)

    @staticmethod
    def _draw_circle_perimeter(img, y, x, rad, white=False):
        """Draws the perimeter of a circle in the given image img, centered in the position (x,y), with radius rad."""
        rr, cc = draw.circle_perimeter(y, x, rad)
        img[rr, cc, :] = white

    @staticmethod
    def _draw_circle(img, y, x, rad, white=False):
        """Draws a circle in the given image img, centered in the position (x,y), with radius rad."""
        rr, cc = draw.disk((y, x), rad)
        img[rr, cc, :] = white

    @staticmethod
    def _draw_point(img, y, x, white=False):
        """Draws a point in the given image img, in the position (x,y)."""
        rr, cc = ([y], [x])
        img[rr, cc, :] = white

    def _draw_wheel(self, img, x, y, rad):
        """Draws a wheel with radius rad in the given image img, in the position (x,y)."""
        if rad > 3:
            self._draw_circle(img, y, x, rad)
            self._draw_circle_perimeter(img, y, x, rad // 2, white=True)
            self._draw_point(img, y, x, white=True)

        elif rad == 3:
            rr, cc = draw.disk((y, x), rad)
            rr = np.delete(rr, [0, 4, len(rr)-5, len(rr)-1])
            cc = np.delete(cc, [0, 4, len(cc)-5, len(cc)-1])
            img[rr, cc, :] = 0
            self._draw_circle_perimeter(img, y, x, rad // 2, white=True)

        elif rad == 2:
            self._draw_circle(img, y, x, rad)
            self._draw_point(img, y, x, white=True)

    @staticmethod
    def _find_next(img, y, x, white=False):
        """Finds the wall of the next car, w.r.t. to the position (x,y), in the image img."""
        curr_x = x
        while img[y, curr_x, 0] != white:
            curr_x = curr_x + 1
        return curr_x

    @staticmethod
    def _find_before(img, y, x, white=False):
        """Finds the wall of the previous car, w.r.t. to the position (x,y), in the image img."""
        curr_x = x
        while img[y, curr_x, 0] != white:
            curr_x = curr_x - 1
        return curr_x

    def _draw_linker(self, img, is_first, is_last):
        """Draws a train coupler in the given image img."""
        y = self.train_config['linker_y']
        edge = self.train_config['half_space']

        if not is_first:
            end = self._find_next(img, y, edge)
            img[y, :end, :] = 0
        if not is_last:
            start = self._find_before(img, y, -edge)
            img[y, start:, :] = 0

    def _render_locomotive(self, wid, hei):
        """"Renders a locomotive in an image with width wid and height hei, the rendered locomotive is appended to the
         list of car images."""
        img = np.ones((hei, wid, 3))

        roof_hei = self.train_config['roof_height']
        y = self.train_config['wheel_base'] + roof_hei + 1

        # draw the locomotive's walls
        rr, cc = draw.polygon_perimeter(
            (y-1, y//2 + roof_hei, y//2 + roof_hei, roof_hei, roof_hei, y-1),
            (0, 0, wid-8, wid-8, wid-2, wid-2))
        img[rr, cc, :] = 0

        # draw the locomotive's front wheel
        x = 2
        self._draw_wheel(img, x, y, 2)

        # draw the locomotive's chimney
        rr, cc = draw.polygon_perimeter(
            (y//2 + roof_hei, y//4 + roof_hei, y//4 + roof_hei, y//2 + roof_hei),
            (x-1, x-1, x+1, x+1))
        img[rr, cc, :] = 0

        # draw the locomotive's rear wheel
        x = x + 6
        y = y - 1
        self._draw_wheel(img, x, y, 3)

        # draw the locomotive's window
        rr, cc = draw.polygon_perimeter(
            (y//2 + roof_hei, y//4 + roof_hei, y//4 + roof_hei, y//2 + roof_hei),
            (wid-6, wid-6, wid-4, wid-4))
        img[rr, cc, :] = 0

        # draw smoke coming out of the locomotive's chimney
        rr, cc = draw.rectangle((y//4 + roof_hei - 3, 1), (y//4 + roof_hei - 2, 2))
        img[rr, cc, :] = 0

        rr, cc = ([y//4 + roof_hei - 4], [3])
        img[rr, cc, :] = 0

        # draw the locomotive's coupler
        self._draw_linker(img, is_first=True, is_last=False)

        self.car_imgs.append(img)

    def _open_square(self, img):
        """Draws an 'open square' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']

        img[roof_hei:hei, edge, :] = 0
        if self.rng_n.random() < THICK_PROB:
            img[roof_hei:hei+1, edge-1, :] = 0

        img[roof_hei:hei, -edge-1, :] = 0
        if self.rng_n.random() < THICK_PROB:
            img[roof_hei:hei+1, -edge, :] = 0

        img[hei, edge:-edge:] = 0
        if self.rng_n.random() < THICK_PROB:
            img[hei+1, edge-1:-edge+1:] = 0

    @staticmethod
    def _draw_ellipse_perimeter(img, r, c, rrad, crad, white=False):
        """Draws the perimeter of an ellipse in the image img."""
        rr, cc = draw.ellipse_perimeter(r, c, rrad, crad)
        img[rr, cc, :] = white

    def _ellipse(self, img):
        """Draws an 'ellipse' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']
        r = hei//2
        crad = img.shape[1]//2 - edge
        c = img.shape[1]//2

        self._draw_ellipse_perimeter(img, r, c, r, crad)
        if self.rng_n.random() < THICK_PROB:
            self._draw_ellipse_perimeter(img, r, c, r, crad - 1)

    @staticmethod
    def _draw_line(img, start, end, white=False):
        """Draws a line, from start to end, in the image img."""
        rr, cc = draw.line(start[1], start[0], end[1], end[0])
        img[rr, cc, :] = white

    def _trapezoid(self, img):
        """Draws a 'trapezoid' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']
        wid = img.shape[1]

        self._draw_line(img, (wid // 5 + edge, hei), (4 * wid // 5 - edge - 1, hei))
        if self.rng_n.random() < THICK_PROB:
            self._draw_line(img, (wid // 5 + edge, hei + 1), (4 * wid // 5 - edge - 1, hei + 1))

        self._draw_line(img, (edge, roof_hei), (wid // 5 + edge, hei))
        if self.rng_n.random() < THICK_PROB:
            self._draw_line(img, (edge + 1, roof_hei), (wid // 5 + edge + 1, hei))

        self._draw_line(img, (wid - edge - 1, roof_hei), (4 * wid // 5 - edge - 1, hei))
        if self.rng_n.random() < THICK_PROB:
            self._draw_line(img, (wid - edge - 2, roof_hei), (4 * wid // 5 - edge - 2, hei))

    @staticmethod
    def _draw_bezier_curve(img, start, middle, end, weight, white=False):
        """Draws a bezier curve in the image img."""
        rr, cc = draw.bezier_curve(start[1], start[0], middle[1], middle[0], end[1], end[0], weight=weight)
        img[rr, cc, :] = white

    def _superellipse(self, img):
        """Draws a 'superellipse' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']
        wid = img.shape[1]

        self._draw_bezier_curve(img, (edge, roof_hei), (edge, hei), (hei + edge, hei), 3.0)
        if self.rng_n.random() < THICK_PROB:
            self._draw_bezier_curve(img, (edge + 1, roof_hei), (edge + 1, hei), (hei + edge, hei - 1), 3.0)

        self._draw_bezier_curve(
            img, (wid - edge - 1, roof_hei), (wid - edge - 1, hei), (wid - edge - hei - 1, hei), 3.0)
        if self.rng_n.random() < THICK_PROB:
            self._draw_bezier_curve(
                img, (wid - edge - 2, roof_hei), (wid - edge - 2, hei), (wid - edge - hei - 1, hei - 1), 3.0)

        self._draw_line(img, (hei + edge, hei), (wid - edge - hei - 1, hei))
        if self.rng_n.random() < THICK_PROB:
            self._draw_line(img, (hei + edge, hei - 1), (wid - edge - hei - 1, hei - 1))

    @staticmethod
    def _draw_polygon_perimeter(img, points, white=False):
        """Draws the perimeter of a polygon with the given points in the image img."""
        xs, ys = ([], [])
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
        rr, cc = draw.polygon_perimeter(ys, xs)
        img[rr, cc, :] = white

    def _hexagon(self, img):
        """Draws a 'hexagon' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']
        wid = img.shape[1]
        if hei % 2 == 0:
            hei += 1

        self._draw_polygon_perimeter(img,
                                     [(edge, (hei+roof_hei) // 2),
                                      (wid//4, roof_hei),
                                      (3*wid//4, roof_hei),
                                      (wid-edge-1, (hei+roof_hei)//2),
                                      (3*wid//4, hei),
                                      (wid//4, hei)])
        if self.rng_n.random() < THICK_PROB:
            self._draw_polygon_perimeter(img,
                                         [(edge+1, (hei+roof_hei) // 2),
                                          (wid//4, roof_hei+1),
                                          (3*wid//4, roof_hei+1),
                                          (wid-edge-2, (hei+roof_hei) // 2),
                                          (3*wid//4, hei-1),
                                          (wid//4, hei-1)])

    def _reinforced(self, img):
        """Draws a 'reinforced' car in the image img."""
        roof_hei = self.train_config['roof_height']
        hei = self.train_config['wheel_base'] + roof_hei
        edge = self.train_config['half_space']

        img[roof_hei:hei, edge, :] = 0
        if self.rng_n.random() < THICK_PROB:
            img[roof_hei:hei+1, edge-1, :] = 0
        img[roof_hei:hei-2, edge+2, :] = 0

        img[roof_hei:hei, -edge-1, :] = 0
        if self.rng_n.random() < THICK_PROB:
            img[roof_hei:hei+1, -edge, :] = 0
        img[roof_hei:hei-2, -edge-3, :] = 0

        img[hei, edge:-edge:] = 0
        if self.rng_n.random() < THICK_PROB:
            img[hei+1, edge-1:-edge+1:] = 0
        img[hei-2, edge+2:-edge-2:] = 0

    def _draw_car(self, img, shape, is_first, is_last):
        """Draws the walls of a car with the given shape shape in the given image img."""
        if shape == 'box':
            self._open_square(img)
        elif shape == 'ellipse':
            self._ellipse(img)
        elif shape == 'trapezoid':
            self._trapezoid(img)
        elif shape == 'superellipse':
            self._superellipse(img)
        elif shape == 'hexagon':
            self._hexagon(img)
        elif shape == 'reinforced':
            self._reinforced(img)

        self._draw_linker(img, is_first, is_last)

    @staticmethod
    def _add_pos_box(add_list, top, bottom, left, right):
        """Appends to add_list a position box with the specified boundaries - top, bottom, left, right."""
        add_list.append(((left, top), (right, top), (right, bottom), (left, bottom)))

    def _draw_wheels(self, img, num_wheels):
        """Draws num_wheels wheels in the given image img."""
        y = self.train_config['wheel_base'] + self.train_config['roof_height'] + 1
        rad = self.train_config['wheel_radius']
        step = img.shape[1] // (num_wheels+1)
        for ix in range(1, num_wheels+1):
            self._draw_wheel(img, ix * step, y, rad)
        self._add_pos_box(self.train_config['wheels_pos'],
                          top=y-rad, bottom=y+rad, left=step-rad, right=num_wheels*step+rad)

    def _add_cargo_pos_box(self, add_list, x, y, rad, x_mul=1, y_mul=1):
        """Adds to add_list a position box with the specified boundaries, centered in (x,y) with radius rad."""
        top = y - y_mul * rad - 1
        bottom = y + y_mul * rad + 1
        left = x - x_mul * rad - 1
        right = x + x_mul * rad + 1
        self._add_pos_box(add_list, top=top, bottom=bottom, left=left, right=right)

    def _draw_cargo_item(self, img, cargo_type, x, y, rad):
        """Draws a cargo of the given cargo type cargo_type in the given image img, in the position (x,y)."""
        if cargo_type == 'ssquare':
            rr, cc = draw.polygon_perimeter((y-rad, y+rad, y+rad, y-rad), (x-rad, x-rad, x+rad, x+rad))
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad)
        elif cargo_type == 'msquare':
            rr, cc = draw.polygon_perimeter((y-rad, y+rad, y+rad, y-rad), (x-2*rad, x-2*rad, x+2*rad, x+2*rad))
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad, x_mul=2)
        elif cargo_type == 'xlsquare':
            rr, cc = draw.polygon_perimeter((y-rad, y+rad, y+rad, y-rad), (x-4*rad, x-4*rad, x+4*rad, x+4*rad))
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad, x_mul=4)
        elif cargo_type == 'circle':
            rr, cc = draw.circle_perimeter(y, x, rad)
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad)
        elif cargo_type == 'triangle':
            rr, cc = draw.polygon_perimeter([y-rad, y+rad, y+rad], [x, x+rad, x-rad])
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad)
        elif cargo_type == 'invtriangle':
            rr, cc = draw.polygon_perimeter([y+rad, y-rad, y-rad], [x, x+rad, x-rad])
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad)
        elif cargo_type == 'diamond':
            rr, cc = draw.polygon_perimeter([y+rad, y, y-rad, y], [x, x+rad, x, x-rad])
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad)
            if rad == 1:
                rr = np.append(rr, [y])
                cc = np.append(cc, [x])
        elif cargo_type == 'hexagon':
            rr, cc = draw.polygon_perimeter(
                [y, y+rad, y+rad, y, y-rad, y-rad],
                [x - 2*rad, x-rad, x+rad, x + 2*rad, x+rad, x-rad])
            self._add_cargo_pos_box(self.train_config['cargos_pos'], x, y, rad, x_mul=2)
        else:  # should never happen, don't do anything
            rr, cc = ([], [])
        img[rr, cc, :] = 0

    def _draw_cargo(self, img, cargo_type, cargo_num, edge, rad, y, car_num):
        """Draws the amount cargo_num of cargo with the given cargo type cargo_type in the given image img."""
        if cargo_num < 2:
            self._draw_cargo_item(img, cargo_type, img.shape[1] // 2, y, rad)
            self.train_config['cargo_belonging'].append(car_num)
        else:
            step = (img.shape[1] - 2*(edge+rad))//(cargo_num-1)
            for ix in range(cargo_num):
                self._draw_cargo_item(img, cargo_type, ix * step + rad + edge, y, rad)
                self.train_config['cargo_belonging'].append(car_num)

    def _draw_roof(self, img, shape, car_shape):
        """Draws a roof with the given shape shape in the given image img."""
        roof_hei = self.train_config['roof_height']
        edge = self.train_config['half_space']
        wid = img.shape[1]

        if shape == 'flat':
            img[roof_hei, edge:-edge, :] = 0
            self._add_pos_box(self.train_config['roofs_pos'],
                              top=roof_hei - 1, bottom=roof_hei + 1, left=edge - 1, right=wid - edge)
        elif shape == 'triangular':
            rr, cc = draw.line(0, wid//2, roof_hei, edge-1)
            img[rr, cc, :] = 0
            rr, cc = draw.line(0, wid//2, roof_hei, wid-edge)
            img[rr, cc, :] = 0
            self._add_pos_box(self.train_config['roofs_pos'],
                              top=0, bottom=roof_hei + 1, left=edge - 2, right=wid - edge + 1)
        elif shape == 'saw':
            step = (wid - 2*edge) // 10
            num = (wid - 2*edge)//step - 1
            for i in range(0, num, 2):
                rr, cc = draw.line(roof_hei, edge + i*step, 0, edge + (i+1)*step)
                img[rr, cc, :] = 0
                rr, cc = draw.line(0, edge + (i+1)*step, roof_hei, edge + (i+2)*step)
                img[rr, cc, :] = 0
            img[roof_hei, -edge, :] = 1
            self._add_pos_box(self.train_config['roofs_pos'],
                              top=0, bottom=roof_hei + 1, left=edge - 1, right=wid - edge)
        elif shape == 'open' and car_shape == 'ellipse':  # ellipse shape is ``closed'', already has a roof
            half_hei = (self.train_config['wheel_base'] + roof_hei) // 2
            self._add_pos_box(self.train_config['roofs_pos'],
                              top=0, bottom=half_hei, left=edge - 1, right=wid - edge)
        elif shape == 'open' and car_shape == 'hexagon':  # hexagon shape is ``closed'', already has a roof
            half_hei = (self.train_config['wheel_base'] + roof_hei) // 2
            self._add_pos_box(self.train_config['roofs_pos'],
                              top=edge, bottom=half_hei + edge, left=edge - 1, right=wid - edge)
        else:  # there is no roof
            self.train_config['roofs_pos'].append(None)

    def _add_noise(self, img, original_shape):
        """Randomly removes pixels from the given image img."""
        num_pixels = img.shape[0] * img.shape[1]
        noise = self.rng_n.integers(0, int(num_pixels * MAX_NOISE) + 1)
        xs = self.rng_n.integers(0, img.shape[0], size=noise)
        ys = self.rng_n.integers(0, img.shape[1], size=noise)

        img[xs, ys, :] = 1

        # num_pixels = original_shape[0] * original_shape[1]
        # noise = self.rng_n.integers(0, int(num_pixels * MAX_NOISE) + 1)
        # xs = self.rng_n.integers(0, original_shape[0], size=noise)
        # ys = self.rng_n.integers(0, original_shape[1], size=noise)

        # valid = (xs < img.shape[0]) & (ys < img.shape[1])
        #
        # xs_valid = xs[valid]
        # ys_valid = ys[valid]
        #
        # img[xs_valid, ys_valid, :] = 1

    def _render_car(self, car, car_num, is_first=False, is_last=False):
        """Renders acar['img_shape'] given car based on its configuration, the rendered car is appended to the list of car images."""
        base_width = self.train_config['base_width']
        base_height = self.train_config['base_height']
        car_image = np.ones((int(base_height * car['height']), int(base_width * car['width']), 3))

        self._draw_car(car_image, car['shape'], is_first, is_last)
        self._draw_wheels(car_image, car['wheels'])
        if car['cargo_type'] != 'none':
            self._draw_cargo(car_image, car['cargo_type'], car['cargo_num'], car['cargo_edge'],
                             car['cargo_rad'], car['cargo_y'], car_num)
        else:
            self.train_config['cargos_pos'].append(None)
            self.train_config['cargo_belonging'].append(car_num)

        self._draw_roof(car_image, car['roof'], car['shape'])

        self._add_noise(car_image, car['img_shape'])

        self.car_imgs.append(car_image)

    @staticmethod
    def _clip_boxes(to_clip, clipping_boxes, clip_to=None):
        """Returns the given boxes to_clip after clipping them to the clipping_boxes, i.e. each to_clip box is summed
         with its corresponding clipping box. The optional argument clip_to allows to manually define which boxes are
          clipped to which clipping boxes."""
        res = []
        if clip_to is None:
            clipping_points = np.asarray([box[0] for box in clipping_boxes])
        else:
            clipping_points = np.asarray([clipping_boxes[box_num][0] for box_num in clip_to])

        for ind, box in enumerate(to_clip):
            if box is None:
                res.append(None)
            else:
                res.append(np.asarray(box) + clipping_points[ind])

        # make list of new points
        temp = res.copy()
        res = []
        for box in temp:
            if box is None:
                res.append(None)
            else:
                new_box = []
                for p in box:
                    new_box.append(tuple(p))
                res.append(tuple(new_box))

        return res

    @staticmethod
    def _after_rotation(points, angle, c):
        """Returns points after a rotation of theta degrees around  point c."""
        m = np.asarray(points)

        x = m[:, 0].copy()
        y = m[:, 1].copy()
        m[:, 0] = np.round(math.cos(angle) * (x - c[0]) - math.sin(angle) * (y - c[1]) + c[0])
        m[:, 1] = np.round(math.sin(angle) * (x - c[0]) + math.cos(angle) * (y - c[1]) + c[1])

        return m

    @staticmethod
    def _mat_to_tuple(mat):
        """Returns the given matrix as a tuple of tuples."""
        return tuple(tuple(p) for p in mat)

    def _rotate_boxes(self, boxes, angle):
        """Returns the given array of position boxes boxes where each position box is rotated by theta degrees."""
        res = []
        for box in boxes:
            if box is None:
                res.append(None)
            else:
                pos = self._after_rotation(box, -angle * (math.pi / 180), (IMG_WID // 2, IMG_HEI // 2))
                res.append(self._mat_to_tuple(pos))
        return res

    @staticmethod
    def _too_black(arr, threshold=0.9):
        """Returns if the given array arr is in average ``too black'' according to a given threshold."""
        tot = float(np.sum(arr))
        if tot / arr.size > (1 - threshold):
            return False
        else:
            return True

    def render_train(self):
        """Renders an image of train based on its configuration."""
        # keep a record of the width and height of the train
        wid, hei = (0, 0)

        # render the locomotive
        self._render_locomotive(LOCOMOTIVE_WIDTH, LOCOMOTIVE_HEIGHT)

        # acquire the list of car/wagon configurations to render
        cars_list = [c[1] for c in self.train_config['cars']]

        # render each car/wagon
        for ix, car in enumerate(cars_list):
            self._render_car(car, ix, False, ix == len(cars_list) - 1)

        # update the width and height of the train
        for car_img in self.car_imgs:
            hei = max(hei, car_img.shape[0])
            wid += car_img.shape[1]

        # if possible calculate a random translation for the train to be drawn
        x = (IMG_WID-wid) // 2  # half of the available space in x axis
        y = (IMG_HEI-hei) // 2  # half of the available space in y axis
        tx, ty = (0, 0)

        if x * 2 - PADDING > 0:
            delta = abs(x - PADDING)
            tx = int(self.rng_p.uniform(-delta, delta))

        if y * 2 - PADDING > 0:
            delta = abs(y - PADDING)
            ty = int(self.rng_p.uniform(-delta, delta))

        # draw the train by drawing the locomotive and each car next to each other
        wid = 0
        top, bottom, left, right = (0, 0, 0, 0)
        for img in self.car_imgs:
            top = y + ty
            bottom = y + ty + img.shape[0]
            left = x + tx + wid
            right = x + tx + wid + img.shape[1]

            self._add_pos_box(self.train_config['cars_pos'], top, bottom, left, right)

            self.train_img[top: bottom, left:right, :] = img
            wid += img.shape[1]

        # record the position box of the train
        left = x + tx
        self._add_pos_box(self.train_config['train_pos'], top=top, bottom=bottom, left=left, right=right)

        # make all the position boxes absolute (instead of relative to the car)
        self.train_config['cargos_pos'] = self._clip_boxes(
            self.train_config['cargos_pos'], self.train_config['cars_pos'][1:], self.train_config['cargo_belonging'])
        self.train_config['roofs_pos'] = self._clip_boxes(
            self.train_config['roofs_pos'], self.train_config['cars_pos'][1:])
        self.train_config['wheels_pos'] = self._clip_boxes(
            self.train_config['wheels_pos'], self.train_config['cars_pos'][1:])

        # rotate the drawn train and record it in the train config

        self.train_img = transform.rotate(self.train_img, self.train_config['angle'], cval=1)

        # update the position boxes by also rotating them
        self.train_config['train_pos'] = self._rotate_boxes(self.train_config['train_pos'], self.train_config['angle'])
        self.train_config['cars_pos'] = self._rotate_boxes(self.train_config['cars_pos'], self.train_config['angle'])
        self.train_config['cargos_pos'] = self._rotate_boxes(self.train_config['cargos_pos'], self.train_config['angle'])
        self.train_config['roofs_pos'] = self._rotate_boxes(self.train_config['roofs_pos'], self.train_config['angle'])
        self.train_config['wheels_pos'] = self._rotate_boxes(self.train_config['wheels_pos'], self.train_config['angle'])

        # add the background image
        if self.background is not None:
            temp_background = np.ones((IMG_HEI, IMG_WID, 3))
            for i in range(0, BACKGROUND_TRIES):
                r = self.rng_b.integers(0, self.background.shape[0] - IMG_HEI)
                c = self.rng_b.integers(0, self.background.shape[1] - IMG_WID)
                temp_background = img_as_float(self.background[r:r + IMG_HEI, c:c + IMG_WID, :])
                if not self._too_black(temp_background):
                    break
            self.train_img *= temp_background

        return img_as_ubyte(self.train_img)

    @staticmethod
    def _draw_box(box, img):
        """Draws the position box box in the given image img."""
        points = [box[0], box[1], box[2], box[3]]

        for ind in range(4):
            next_ind = (ind + 1) % 4
            rr, cc = draw.line(points[ind][1], points[ind][0], points[next_ind][1], points[next_ind][0])
            img[rr, cc, :] = True

    def draw_position_boxes(self, train_box=True, car_boxes=True, cargo_boxes=True, roof_boxes=True, wheel_boxes=True):
        """Renders the position boxes of the train. Note: Should only be called after `render_train'."""

        if train_box:
            self._draw_box(self.train_config['train_pos'][0], self.train_img)

        if car_boxes:
            for car_box in self.train_config['cars_pos']:
                self._draw_box(car_box, self.train_img)

        if cargo_boxes:
            for cargo_box in self.train_config['cargos_pos']:
                if cargo_box is not None:
                    self._draw_box(cargo_box, self.train_img)

        if roof_boxes:
            for roof_box in self.train_config['roofs_pos']:
                if roof_box is not None:
                    self._draw_box(roof_box, self.train_img)

        if wheel_boxes:
            for wheel_box in self.train_config['wheels_pos']:
                self._draw_box(wheel_box, self.train_img)

        return img_as_ubyte(self.train_img)

if __name__ == "__main__":
    print("a")
