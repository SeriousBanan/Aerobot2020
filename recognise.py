"""
Module to recognise template on images and QR-codes.

How to find picture on image:
    First generate template object using function generate_template.
        template_obj = generate_template(template_path)

    Then use function recognise_picture to recognise template on the
    source image. If template has found it return numpy array with
    coords of edges found template else it return None.
    Use key-arg trace=True to show images with found template.
        dst = recognise_picture(source_image, template_obj)

How to find QR-code:
    Use function recognise_qr giving source image in args.
    If QR has found return data from QR and it's position
    else return None.
"""

from itertools import combinations
from math import isqrt
import numpy
import cv2
from pyzbar import pyzbar

SIFT = None
MATCHER = None
KERNEL = -1 / 256 * numpy.array([[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, -476, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]])


def is_right_form(pts: list, frame_shape: (int, int), *, min_size: int = 25):
    """
    Check is the found quadrilateral is of normal shape and size.
    """
    def make_line(pt1, pt2):
        """
        Create line (y = a*x + b) from two points and return the dict
        with two coefficients: 'a' and 'b'.
        """
        x1, y1 = pt1
        x2, y2 = pt2

        if x1 == x2:
            x1 += 0.001

        return {
            "a": (y1 - y2) / (x1 - x2),
            "b": (y2 * x1 - y1 * x2) / (x1 - x2)
        }

    # Reformate array of points
    pts = [pt[0] for pt in pts]

    # If at least one point is outside the image it is wrong
    if any(x > frame_shape[1] or y > frame_shape[0] for x, y in pts):
        return False

    # ! Если разкоментить меньше ошибок, но реже замечает
    # # If distance betwen two points less than min_size it is wrong
    # if any(isqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) <= min_size
    #        for pt1, pt2 in combinations(pts, r=2)):
    #     return False

    line12 = make_line(pts[0], pts[1])
    line23 = make_line(pts[1], pts[2])
    line34 = make_line(pts[2], pts[3])
    line41 = make_line(pts[3], pts[0])

    # If we have intersections it is wrong
    if (line12["a"] != line34["a"] and (
        min(pts[0][0], pts[1][0]) <=
            (line34["b"] - line12["b"]) / (line12["a"] - line34["a"]) <=
            max(pts[0][0], pts[1][0]) or
        min(pts[2][0], pts[3][0]) <=
            (line34["b"] - line12["b"]) / (line12["a"] - line34["a"]) <=
            max(pts[2][0], pts[3][0])) or
        line23["a"] != line41["a"] and (
        min(pts[1][0], pts[2][0]) <=
            (line23["b"] - line41["b"]) / (line41["a"] - line23["a"]) <=
            max(pts[1][0], pts[2][0]) or
        min(pts[0][0], pts[3][0]) <=
            (line23["b"] - line41["b"]) / (line41["a"] - line23["a"]) <=
            max(pts[0][0], pts[3][0]))):
        return False

    return True


def generate_template(template):
    """
    Generate object with information about template:
    image, keypoints and descriptors
    """
    global SIFT
    global MATCHER

    if SIFT is None:
        SIFT = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
        MATCHER = cv2.BFMatcher()  # BFMatcher with default params

    # If we have path in template we try to open it.
    if isinstance(template, str):
        if template.startswith("rtsp"):
            cap = cv2.VideoCapture(template)
            _, template = cap.read()
            cap.release()
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

    template_obj = {"template": cv2.filter2D(template, -1, KERNEL)}
    template_obj["template_keypoints"], template_obj["template_descriptors"] = SIFT.detectAndCompute(
        template_obj["template"], None)

    return template_obj


def recognise_picture(
        source: str or numpy.ndarray,
        template_obj: dict,
        *,
        min_match_count: int = 18,
        dist_coeff: float = 0.9,
        trace: bool = False) -> None or numpy.ndarray:
    """
    Find template image on source image.
    If source given as str we will try to
    open image, located on this path.
    """
    global SIFT
    global MATCHER

    if SIFT is None:
        SIFT = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
        MATCHER = cv2.BFMatcher()  # BFMatcher with default params

    # If we have path in source we try to open it.
    if isinstance(source, str):
        if source.startswith("rtsp"):
            cap = cv2.VideoCapture(source)
            _, source = cap.read()
            cap.release()
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            source = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

    # rows, cols = source_image.shape
    # matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 14, 1)
    # source_image = cv2.warpAffine(source_image, matrix, (cols, rows))

    source = cv2.filter2D(source, -1, KERNEL)

    source_keypoints, source_descriptors = SIFT.detectAndCompute(source, None)
    try:
        matches = MATCHER.knnMatch(
            template_obj["template_descriptors"], source_descriptors, k=2)
    except BaseException:
        print("[-] что-то не так с точками")
        return None

    if matches and len(matches[0]) == 1:
        print("[-] что-то не так с точками")
        return None

    good = [m for m, n in matches if m.distance < n.distance * dist_coeff]

    if len(good) < min_match_count:
        print(
            f"[-] количество совпадений недостаточно - {len(good)}/{min_match_count}")
        return None

    template_points = numpy.float32(
        [template_obj["template_keypoints"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    source_points = numpy.float32(
        [source_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(
        template_points, source_points, cv2.RANSAC, 5.0)

    if matrix is None:
        print("[-] что-то не так с точками")
        return None

    h, w = template_obj["template"].shape[:2]  # размеры шаблона
    points = numpy.asarray([[0, 0], [0, h - 1], [w - 1, h - 1],
                            [w - 1, 0]], dtype=numpy.float32).reshape(-1, 1, 2)

    # выполняем преобразования координат точек рамки шаблона
    dst = cv2.perspectiveTransform(points, matrix)

    # обрезаем рамку вылезшую за пределы картинки
    dst = [numpy.int32(numpy.abs(dst))]

    if not is_right_form(dst[0], source.shape[:2]):
        print("[-] Неправильная форма")
        return None

    if trace:
        # рисуем рамку вокруг найденного объекта
        img_res = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        img_res = cv2.polylines(
            img_res, dst, True, (0, 255, 255), 2, cv2.LINE_AA)

        center_dot = (sum(dot[0] for [dot] in dst[0]) // 4,
                      sum(dot[1] for [dot] in dst[0]) // 4)
        img_res = cv2.circle(img_res, center_dot, 2, (0, 0, 255), 5)

        # рисуем совпадения контрольных точек
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        img_res = cv2.drawMatches(
            template_obj["template"],
            template_obj["template_keypoints"],
            img_res,
            source_keypoints,
            good,
            None,
            **draw_params)

        # Show result
        cv2.imshow("frame", img_res)

    print(f"[+] найдено {len(good)} совпадений")
    return dst[0]


def recognise_qr(source: str or numpy.ndarray, *,
                 trace: bool = False) -> (str, numpy.ndarray) or None:
    """
    Recognise QR codes on pictures and return array of tuples with qr-data
    and points with positions of verties of QR on source or return None if
    QR-code not found.
    """

    if isinstance(source, str):
        if source.startswith("rtsp"):
            cap = cv2.VideoCapture(source)
            _, source = cap.read()
            cap.release()
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            source = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

    barcodes = pyzbar.decode(source)
    res = []
    for barcode in barcodes:
        bbox = numpy.array(
            [[point for point in barcode.polygon]], dtype=numpy.int)
        data = barcode.data.decode("utf-8")
        res.append((data, bbox))

    if not res:
        return None

    print("QR Code detected: ", [data for data, bbox in res])
    if trace:
        for data, bbox in res:
            img_res = cv2.polylines(source, bbox, True, (0, 255, 255))

        cv2.imshow("img", img_res)
    return res
