import numpy as np

from PIL import Image, ImageFilter, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.restoration import inpaint_biharmonic


class BoundingBoxInterventionPrimitives:
    # box (rectangular)
    @staticmethod
    def blur(arr, box, sigma):
        x0, y0, x1, y1 = box
        crop = Image.fromarray(arr[y0:y1, x0:x1])
        out = arr.copy()
        out[y0:y1, x0:x1] = np.array(crop.filter(ImageFilter.GaussianBlur(sigma)))
        return out

    @staticmethod
    def grey(arr, box, level=128):
        x0, y0, x1, y1 = box
        out = arr.copy()
        out[y0:y1, x0:x1] = level
        return out

    @staticmethod
    def invert(arr, box):
        x0, y0, x1, y1 = box
        out = arr.copy()
        out[y0:y1, x0:x1] = 255 - out[y0:y1, x0:x1]
        return out

    # circular
    @staticmethod
    def _circle_mask(box):
        x0, y0, x1, y1 = box
        h, w = y1 - y0, x1 - x0
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = min(h, w) / 2
        yy, xx = np.ogrid[:h, :w]
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

    @staticmethod
    def blur_circle(arr, box, sigma):
        x0, y0, x1, y1 = box
        mask = BoundingBoxInterventionPrimitives._circle_mask(box)
        crop = arr[y0:y1, x0:x1]
        blurred = np.array(Image.fromarray(crop).filter(ImageFilter.GaussianBlur(sigma)))
        out = arr.copy()
        region = out[y0:y1, x0:x1]
        region[mask] = blurred[mask]
        return out

    @staticmethod
    def grey_circle(arr, box, level=128):
        x0, y0, x1, y1 = box
        mask = BoundingBoxInterventionPrimitives._circle_mask(box)
        out = arr.copy()
        region = out[y0:y1, x0:x1]
        region[mask] = level
        return out

    @staticmethod
    def invert_circle(arr, box):
        x0, y0, x1, y1 = box
        mask = BoundingBoxInterventionPrimitives._circle_mask(box)
        out = arr.copy()
        region = out[y0:y1, x0:x1]
        region[mask] = 255 - region[mask]
        return out

    @staticmethod
    def _polygon_mask(poly, shape):
        h, w = shape[:2]
        m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
        return np.array(m, dtype=bool)

    @staticmethod
    def blur_polygon(arr, poly, sigma):
        mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
        blurred = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(sigma)))
        out = arr.copy()
        out[mask] = blurred[mask]
        return out

    @staticmethod
    def grey_polygon(arr, poly, level=128):
        mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
        out = arr.copy()
        out[mask] = level
        return out

    @staticmethod
    def invert_polygon(arr, poly):
        mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
        out = arr.copy()
        out[mask] = 255 - out[mask]
        return out

    @staticmethod
    def context_fill_polygon(arr, poly, ring=6):
        mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
        ring_mask = binary_dilation(mask, iterations=ring) & ~mask  # boundary band
        fill = np.median(arr[ring_mask], axis=0)
        out = arr.copy()
        out[mask] = fill.astype(arr.dtype)
        return out
    #
    # @staticmethod
    # def inpaint_polygon(arr, poly):
    #     mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
    #     filled = inpaint_biharmonic(arr, mask, channel_axis=-1)
    #     if filled.max() <= 1.0:
    #         filled = filled * 255.0
    #     return filled.astype(np.uint8)

    @staticmethod
    def inpaint_polygon(arr, poly, dilate=2):
        mask = BoundingBoxInterventionPrimitives._polygon_mask(poly, arr.shape)
        filled = inpaint_biharmonic(arr, mask, channel_axis=-1, split_into_regions=True)
        if filled.max() <= 1.0:
            filled = filled * 255.0
        return filled.astype(np.uint8)
