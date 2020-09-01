"""
Module to recognise template on images.

How to use:
First generate template object using function generate_template.
    template_obj = generate_template(template_path)

Then use function recognise_picture to recognise template on the
source image. If template has found it return numpy array with
coords of edges found template else it return None.
Use key-arg trace=True to show images with found template.
    dst = recognise_picture(source_image, template_obj)
"""

from itertools import combinations
from math import isqrt
import numpy
import cv2

SIFT = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
MATCHER = cv2.BFMatcher()  # BFMatcher with default params


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

    # If distance betwen two points less than min_size it is wrong
    if any(isqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) <= min_size
           for pt1, pt2 in combinations(pts, r=2)):
        return False

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

    # If we have path in template we try to open it.
    if isinstance(template, str):
        if template.startswith("rtsp"):
            cap = cv2.VideoCapture(template)
            _, template = cap.read()
            cap.release()
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

    template_obj = {"template": template}
    template_obj["template_keypoints"], template_obj["template_descriptors"] = SIFT.detectAndCompute(
        template_obj["template"], None)

    return template_obj


def recognise_picture(
        source: str or numpy.ndarray,
        template_obj: dict,
        *,
        min_match_count: int = 20,
        dist_coeff: float = 0.9,
        trace: bool = False) -> None or numpy.ndarray:
    """
    Find template image on source image.
    If source given as str we will try to
    open image, located on this path.
    """

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

    source_keypoints, source_descriptors = SIFT.detectAndCompute(source, None)
    matches = MATCHER.knnMatch(
        template_obj["template_descriptors"], source_descriptors, k=2)

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

    h, w = template_obj["template"].shape  # размеры шаблона
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


if __name__ == "__main__":
    import sys
    print("OpenCV ", cv2.__version__)

    if len(sys.argv) == 1:
        template_path = "Logo.png"
        video_path = "Sources/video_2020-08-28_15-53-46.mp4"
        out_path = "output.avi"
    elif len(sys.argv) == 3:
        template_path, video_path, out_path = sys.argv[1:3], ""
    elif len(sys.argv) == 4:
        template_path, video_path, out_path = sys.argv[1:4]

    template_obj = generate_template(template_path)

    cap = cv2.VideoCapture(video_path)

    if out_path:
        frame_width = int(cap.get(3) * 0.4)
        frame_height = int(cap.get(4) * 0.4)

        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(
                'M',
                'J',
                'P',
                'G'),
            20,
            (frame_width,
             frame_height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        source_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dsize = (int(source_image.shape[1] * 0.4),
                 int(source_image.shape[0] * 0.4))
        source_image = cv2.resize(
            src=source_image,
            dsize=dsize,
            interpolation=cv2.INTER_CUBIC)

        dst = recognise_picture(source_image, template_obj, trace=True)

        if dst is not None:
            img_res = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
            img_res = cv2.polylines(
                img_res, [dst], True, (0, 255, 255), 2, cv2.LINE_AA)

            # записываем результат
            if out:
                out.write(img_res)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
