from itertools import combinations
from math import isqrt
import numpy as np
import cv2


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


def from_cam():
    min_match_count = 20  # порог минимального количества совпадений ключевых точек
    dist_coeff = 0.9  # 0.65

    template_path = "Logo.png"

    sift = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector

    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #     search_params = dict(checks = 50)nty
    #     matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matcher = cv2.BFMatcher()  # BFMatcher with default params

    # find the keypoints and descriptors of template
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_keypoints, template_descriptors = sift.detectAndCompute(
        template_image, None)

    cap = cv2.VideoCapture("Sources/video_2020-08-28_15-53-46.mp4")

    frame_width = 737
    frame_height = 768

    out = cv2.VideoWriter(
        'video_2020-08-28_15-53-46.avi',
        cv2.VideoWriter_fourcc(
            'M',
            'J',
            'P',
            'G'),
        20,
        (frame_width,
         frame_height))

    import time
    start_time = time.time()
    statistic = {True: 0, False: 0}
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # cv2.imshow("f", ulin.edit_image(frame))
        # frame = ulin.edit_image(frame)

        source_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dsize = (int(source_image.shape[1] * 0.4),
                 int(source_image.shape[0] * 0.4))
        source_image = cv2.resize(
            src=source_image,
            dsize=dsize,
            interpolation=cv2.INTER_CUBIC)

        # rows, cols = source_image.shape
        # matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 14, 1)
        # source_image = cv2.warpAffine(source_image, matrix, (cols, rows))

        source_keypoints, source_descriptors = sift.detectAndCompute(
            source_image, None)
        matches = matcher.knnMatch(
            template_descriptors, source_descriptors, k=2)

        good = [m for m, n in matches if m.distance < n.distance * dist_coeff]

        if len(good) < min_match_count:
            statistic[False] += 1
            print(
                "[-] количество совпадений недостаточно - %d/%d" %
                (len(good), min_match_count))
        else:

            template_points = np.float32(
                [template_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            source_points = np.float32(
                [source_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(
                template_points, source_points, cv2.RANSAC, 5.0)

            h, w = template_image.shape  # размеры шаблона
            points = np.asarray([[0, 0], [0, h - 1], [w - 1, h - 1],
                                 [w - 1, 0]], dtype=np.float32).reshape(-1, 1, 2)

            # выполняем преобразования координат точек рамки шаблона
            if matrix is None:
                statistic[False] += 1
                continue

            dst = cv2.perspectiveTransform(points, matrix)

            # обрезаем рамку вылезшую за пределы картинки
            dst = [np.int32(np.abs(dst))]

            # рисуем рамку вокруг найденного объекта
            img_res = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
            img_res = cv2.polylines(
                img_res, dst, True, (0, 255, 255), 2, cv2.LINE_AA)

            img_res = cv2.polylines(
                img_res, np.array([[0, 0], [25, 0], [25, 25], [0, 25]],
                                  np.int32).reshape((-1, 1, 2)), True, (0, 0, 255), 2, cv2.LINE_AA)
            img_res = cv2.circle(
                img_res, (dst[0][0][0][0], dst[0][0][0][1]), 1, (255, 0, 0), 5)
            img_res = cv2.circle(
                img_res, (dst[0][1][0][0], dst[0][1][0][1]), 1, (255, 255, 0), 5)

            if not is_right_form(dst[0], img_res.shape[:2]):
                statistic[False] += 1
                print("[-] Неправильная форма")
                continue

            # рисуем совпадения контрольных точек
            matches_mask = mask.ravel().tolist()
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matches_mask,  # draw only inliers
                               flags=2)
            img_res = cv2.drawMatches(
                template_image,
                template_keypoints,
                img_res,
                source_keypoints,
                good,
                None,
                **draw_params)

            # записываем результат
            cv2.imshow("frame", img_res)
            out.write(img_res)

            statistic[True] += 1
            print("[+] найдено %d совпадений" % len(good))

        if cv2.waitKey(1) == ord("q"):
            break
    print(
        f"Average time for detection: {(time.time() - start_time) / (statistic[True] + statistic[False])}\n"
        f"True-positives: {statistic[True] / (statistic[True] + statistic[False])}\n"
        f"False-negatives: {statistic[False] / (statistic[True] + statistic[False])}\n")

    cap.release()
    out.release()


if __name__ == "__main__":
    print("OpenCV ", cv2.__version__)

    from_cam()
