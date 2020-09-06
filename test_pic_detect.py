import cv2
import numpy
from recognise import *


def check_recognise_no_clastering():

    img1 = cv2.imread(r"Sources\4r7MyVVknms.jpg")
    img2 = cv2.imread(r"Sources\CsaKsshG1uI.jpg")
    img3 = cv2.imread(r"Sources\Q_9a6AGvAlo.jpg")

    template_path = "Logo.png"
    template = cv2.imread(template_path, cv2.COLOR_BGR2GRAY)
    template_obj = recognise.generate_template(template_path)

    for img in img1, img2, img3:
        source_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", source_image)

        dst = recognise.recognise_picture(
            source_image, template_obj, trace=True)
        cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    print("OpenCV ", cv2.__version__)

    if len(sys.argv) == 1:
        template_path = "Logo.png"
        video_path = "Sources/Video2.mp4"
        out_path = "output2.avi"
    elif len(sys.argv) == 3:
        template_path, video_path, out_path = sys.argv[1:3], ""
    elif len(sys.argv) == 4:
        template_path, video_path, out_path = sys.argv[1:4]

    template_obj = generate_template(template_path)

    cap = cv2.VideoCapture(video_path)

    if out_path:
        frame_width = int(cap.get(3) * 0.4 / 0.4)
        frame_height = int(cap.get(4) * 0.4 / 0.4)

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
        dsize = (int(source_image.shape[1] * 0.4 / 0.4),
                 int(source_image.shape[0] * 0.4 / 0.4))
        source_image = cv2.resize(
            src=source_image,
            dsize=dsize,
            interpolation=cv2.INTER_CUBIC)

        dst = recognise_picture(
            source_image,
            template_obj,
            trace=True,
            min_match_count=18)

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
