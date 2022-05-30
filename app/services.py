"""Функции для обработки изображения"""
import os
import cv2
import urllib.request
import numpy as np
import base64
import seaborn as sns

from matplotlib import pyplot as plt




# define the path to the face detector and smile detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

SMILE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_smile.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

# path to trained faces and labels
TRAINED_FACES_PATH = "{base_path}/faces".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

# maximum distance between face and match
THRESHOLD = 75

# create the cascade classifiers
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
smiledetector = cv2.CascadeClassifier(SMILE_DETECTOR_PATH)


def _grab_image(path=None, base64_string=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
                image = np.asarray(bytearray(data), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # if the stream is not None, then the image has been uploaded
        elif base64_string is not None:
            # sbuf = StringIO()
            # sbuf.write(base64.b64decode(base64_string))
            # pimg = Image.open(sbuf)
            # image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

            image = base64.b64decode(base64_string)
            image = np.fromstring(image, dtype=np.uint8)
            image = cv2.imdecode(image, 1)
    # convert the image to a NumPy array and then read it into
    # OpenCV format
    # return the image
    return image


# laplacian
def get_variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


# findconturs
def get_conturs(img, gray_image):
    # Контуры
    edged = cv2.Canny(gray_image, 10, 300)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("Number of Contours found = " + str(len(contours)))
    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
    cv2.drawContours(gray_image, contours, -1, (0, 255, 255), 1)
    # viewImage(img)
    return int(len(contours))


# draw lines of composition
def draw_lines_of_composition(img):
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    # линии
    cv2.line(img, (int(width / 3), 0), (int(width / 3), height), (200, 255, 255), 1)
    cv2.line(img, (int(width * 2 / 3), 0), (int(width * 2 / 3), height), (200, 255, 255), 1)
    cv2.line(img, (0, int(height / 3)), (int(width), int(height / 3)), (200, 255, 255), 1)
    cv2.line(img, (0, int(height * 2 / 3)), (int(width), int(height * 2 / 3)), (200, 255, 255), 1)

    cv2.circle(img, (int(width / 3), int(height / 3)), 5, (0, 255, 100), -1)
    cv2.circle(img, (int(width * 2 / 3), int(height / 3)), 5, (0, 255, 100), -1)
    cv2.circle(img, (int(width / 3), int(height * 2 / 3)), 5, (0, 255, 100), -1)
    cv2.circle(img, (int(width * 2 / 3), int(height * 2 / 3)), 5, (0, 255, 100), -1)

    cv2.line(img, (int(width / 3), int(height / 3)), (int(width / 3), int(height * 2 / 3)), (0, 255, 100), 2)
    cv2.line(img, (int(width / 3), int(height / 3)), (int(width * 2 / 3), int(height / 3)), (0, 255, 100), 2)
    cv2.line(img, (int(width / 3), int(height * 2 / 3)), (int(width * 2 / 3), int(height * 2 / 3)), (0, 255, 100), 2)
    cv2.line(img, (int(width * 2 / 3), int(height / 3)), (int(width * 2 / 3), int(height * 2 / 3)), (0, 255, 100), 2)

    wh = int(width / 6)
    cv2.circle(img, (int(width / 3), int(height / 3)), wh, (0, 255, 100), 1)
    cv2.circle(img, (int(width * 2 / 3), int(height / 3)), wh, (0, 255, 100), 1)
    cv2.circle(img, (int(width / 3), int(height * 2 / 3)), wh, (0, 255, 100), 1)
    cv2.circle(img, (int(width * 2 / 3), int(height * 2 / 3)), wh, (0, 255, 100), 1)


# print blur/focus level
def print_blur_image(img, iMax):
    a = list(range(iMax))  # список по количеству строк
    for iW in range(0, iMax, 1):
        a[iW] = list(range(iMax))
        scale_percent = 100
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        for iH in range(0, iMax, 1):
            crop_img = img[int(height * iH / iMax):int(height * (iH + 1) / iMax),
                       int(width * iW / iMax):int(width * (iW + 1) / iMax)]
            icrop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            a[iW][iH] = int(get_variance_of_laplacian(icrop))

    for iW in range(0, iMax, 1):
        for iH in range(0, iMax, 1):
            if a[iH][iW] < np.mean(a):

                cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
                cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                         (int(width * iW / iMax), int(height * iMax / iMax)), (100, 100, 100), 1)
                cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                         (int(width * iMax / iMax), int(height * iH / iMax)), (100, 100, 100), 1)
            else:
                if a[iH][iW] < np.max(a) * 0.5:
                    cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 230), 2)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iW / iMax), int(height * iMax / iMax)), (200, 100, 255), 1)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iMax / iMax), int(height * iH / iMax)), (200, 100, 255), 1)
                else:  # max
                    cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iW / iMax), int(height * iMax / iMax)), (0, 0, 255), 1)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iMax / iMax), int(height * iH / iMax)), (0, 0, 255), 1)
    return a


# Контурный анализ
def print_conturs_of_image(img, iMax):
    a = list(range(iMax))  # список по количеству строк
    for iW in range(0, iMax, 1):
        a[iW] = list(range(iMax))
        scale_percent = 100
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        for iH in range(0, iMax, 1):
            crop_img = img[int(height * iH / iMax):int(height * (iH + 1) / iMax),
                       int(width * iW / iMax):int(width * (iW + 1) / iMax)]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            icrop = cv2.Canny(crop_img, 5, 300)
            contours, hierarchy = cv2.findContours(icrop,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            a[iW][iH] = int(str(len(contours)))
            # cv2.drawContours(img, contours, -1, (0, 255, 255), 1)

    for iW in range(0, iMax, 1):
        for iH in range(0, iMax, 1):

            if a[iH][iW] < np.mean(a):
                cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                         (int(width * iW / iMax), int(height * iMax / iMax)), (200, 255, 255), 1)
                cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                         (int(width * iMax / iMax), int(height * iH / iMax)), (200, 255, 255), 1)
            else:
                if a[iH][iW] < int(np.max(a) * 0.5):
                    h1 = int(height * iH / iMax)
                    h2 = int(height * (iH + 1) / iMax)
                    w1 = int(width * iW / iMax)
                    w2 = int(width * (iW + 1) / iMax)

                    cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 230), 1)  # 255, 215, 0
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iW / iMax), int(height * iMax / iMax)), (200, 255, 255), 1)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iMax / iMax), int(height * iH / iMax)), (200, 255, 255), 1)
                else:  # max
                    cv2.putText(img, str(a[iH][iW]), (int(width * iW / iMax) + 10, int(height * iH / iMax) + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iW / iMax), int(height * iMax / iMax)), (220, 200, 0), 1)
                    cv2.line(img, (int(width * iW / iMax), int(height * iH / iMax)),
                             (int(width * iMax / iMax), int(height * iH / iMax)), (220, 200, 0), 1)
    return a


def draw_histogram(image, filename):
    image_blue = image[:, :, 0]
    image_green = image[:, :, 1]
    image_red = image[:, :, 2]

    X_b = np.array(image_blue.flatten())[:, np.newaxis]
    X_g = np.array(image_green.flatten())[:, np.newaxis]
    X_r = np.array(image_red.flatten())[:, np.newaxis]

    # print("Изображение успешно загружено и разделено на каналы!")
    fig, (RGB, blue, green, red) = plt.subplots(4, 1)
    fig.set_size_inches(18.5, 10.5)

    RGB.legend(loc='upper right')
    RGB.set_title('Гистограмма RGB:', fontsize=12)
    sns.distplot(X_b, bins=256, kde=False, rug=False, norm_hist=False, ax=RGB, color="blue", label="Синий Канал")
    sns.distplot(X_g, bins=256, kde=False, rug=False, norm_hist=False, ax=RGB, color="green", label="Зеленый Канал")
    sns.distplot(X_r, bins=256, kde=False, rug=False, norm_hist=False, ax=RGB, color="red", label="Красный канал")
    RGB.legend(loc='upper right')

    blue.set_title('Синий Канал:', fontsize=12)
    sns.distplot(X_b, bins=256, kde=False, rug=False, norm_hist=False, ax=blue, color="blue", label="Синий Канал")
    blue.legend(loc='upper right')

    green.set_title('Зеленый Канал:', fontsize=12)
    sns.distplot(X_g, bins=256, kde=False, rug=False, norm_hist=False, ax=green, color="green", label="Зеленый Канал")
    green.legend(loc='upper right')

    red.set_title('Красный канал:', fontsize=12)
    sns.distplot(X_r, bins=256, kde=False, rug=False, norm_hist=False, ax=red, color="red", label="Красный канал")
    red.legend(loc='upper right')

    plt.savefig(TRAINED_FACES_PATH + "/" + "img/2/" + filename)

    # Экстремумы
    bandwidth = 10

    X_plot = np.linspace(0, 255, 256)[:, np.newaxis]
    # print(type(X_plot))
    # print("Загружаем данные о цветовых каналах в модель. Построение модели происходит для каждого канала в отдельности")
    kde_blue = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(X_b)
    # print("Аппроксимация для синего канала успешно рассчитана!")
    kde_green = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(X_g)
    # print("Аппроксимация для зеленого канала успешно рассчитана!")
    kde_red = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(X_r)
    # print("Аппроксимация для красного канала успешно рассчитана!")
    log_dens_blue = kde_blue.score_samples(X_plot)
    log_dens_green = kde_green.score_samples(X_plot)
    log_dens_red = kde_red.score_samples(X_plot)

    fig_2, (blue_kde, green_kde, red_kde) = plt.subplots(3, 1)
    fig_2.set_size_inches(18.5, 10.5)

    sns.distplot(X_b, bins=256, kde=False, rug=False, norm_hist=True, ax=blue_kde, color="blue", label="Blue Channel")
    sns.distplot(X_g, bins=256, kde=False, rug=False, norm_hist=True, ax=green_kde, color="green",
                 label="Green Channel")
    sns.distplot(X_r, bins=256, kde=False, rug=False, norm_hist=True, ax=red_kde, color="red", label="Red Channel")
    blue_kde.plot(X_plot[:, 0], np.exp(log_dens_blue), '-', label="Kernel - Epanechnikov")
    green_kde.plot(X_plot[:, 0], np.exp(log_dens_green), '-', label="Kernel - Epanechnikov")
    red_kde.plot(X_plot[:, 0], np.exp(log_dens_red), '-', label="Kernel - Epanechnikov")

    blue_kde.legend(loc='upper right')
    green_kde.legend(loc='upper right')
    red_kde.legend(loc='upper right')

    fig_3, (diff_ax) = plt.subplots(1, 1)
    fig_3.set_size_inches(18.5, 10.5)

    ################################### Рассчет первой производной ######################################
    diff = np.gradient(np.exp(log_dens_blue))
    diff_list = list(diff)
    xc_1, xi_1 = pyaC.zerocross1d(X_plot[:, 0], diff, getIndices=True)

    # zeros_list = list(int(x) for x in xc_1)
    zeros = np.array(np.reshape(xc_1, (len(xc_1), 1)))
    samples = kde_blue.score_samples(zeros)
    log_dens_blue_in_zeros = np.exp(samples)
    # построение графика функции аппроксимации и первой производной #
    diff_ax.set_ylim(-0.005, 0.025)
    diff_ax.axhline(y=0, color="black")
    diff_ax.set_title('First derivative over KDE Function (Epanechnikov):', fontsize=12)
    diff_ax.plot(X_plot[:, 0], diff, color="red", label="Первая производная от функции аппроксимации")
    diff_ax.scatter(xc_1, log_dens_blue_in_zeros, marker='*', s=130, color="blue", label="Ноль производной - экстремум")
    diff_ax.plot(X_plot[:, 0], np.exp(log_dens_blue), '-', label="Функция аппроксимации с ядром Епанечникова")
    diff_ax.legend(loc='upper right')

    # Проверка принадлежности к интервалу от 0 до 255 и назначение уровня освещенности по координате X_extremum
    # Координаты экстремума:

    m = max(samples)
    index_global_max = [i for i, j in enumerate(samples) if j == m]
    X_extremum = int(xc_1[index_global_max])

    rec = ""
    if X_extremum in range(0, 43):
        rec = "Изображение слишком темное -3exp"
    elif X_extremum in range(43, 85):
        rec = "Изображение весьма темное -2exp"
    elif X_extremum in range(85, 128):
        rec = "Изображение немного темное -1exp"
    elif X_extremum in range(128, 170):
        rec = "Изображение немного светлое +1exp"
    elif X_extremum in range(170, 212):
        rec = "Изображение весьма светлое +2exp"
    elif X_extremum in range(212, 255):
        rec = "Изображение слишком светлое +3exp"

    #

    plt.savefig(TRAINED_FACES_PATH + "/" + "img/4/" + filename)

    return X_extremum, rec


def analyze_photo(cat, filename, id_tut):
    url = cat + filename

    image = _grab_image(url=url)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = get_variance_of_laplacian(gray_img)

    faces = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30), flags=0)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # рисуем линии
    draw_lines_of_composition(image)
    # композиция
    cv2.imwrite(TRAINED_FACES_PATH + "/" + "img/1/" + filename, image)

    image = _grab_image(url=url)
    # фокус лапласиан
    a = print_blur_image(image, 9)
    # focus
    iMax = 9
    sum = 0
    for iW in range(0, iMax, 1):
        for iH in range(0, iMax, 1):
            if iH == 0 or iH == 8 or iW == 0 or iW == 8:
                a[iH][iW] = int(a[iH][iW] * 0.25)
            elif iH == 1 or iH == 7 or iW == 1 or iW == 7:
                a[iH][iW] = int(a[iH][iW] * 0.5)
            elif iH == 2 or iH == 6 or iW == 2 or iW == 6:
                a[iH][iW] = int(a[iH][iW] * 0.75)
            sum = sum + a[iH][iW]
    fcs = 0.5 * (int(fm) + int(sum / (iMax * iMax)))

    cv2.imwrite(TRAINED_FACES_PATH + "/" + "img/0/" + filename, image)
    cv2.imwrite(TRAINED_FACES_PATH + "/" + "img/5/" + filename, gray_img)

    # фокус конутрный
    imgcn = _grab_image(url=url)
    gray = cv2.cvtColor(imgcn, cv2.COLOR_BGR2GRAY)
    cnt_0 = get_conturs(imgcn, gray)
    a = print_conturs_of_image(imgcn, 9)

    sum = 0
    for iW in range(0, iMax, 1):
        for iH in range(0, iMax, 1):
            if iH == 0 or iH == 8 or iW == 0 or iW == 8:
                a[iH][iW] = int(a[iH][iW] * 0.25)
            elif iH == 1 or iH == 7 or iW == 1 or iW == 7:
                a[iH][iW] = int(a[iH][iW] * 0.5)
            elif iH == 2 or iH == 6 or iW == 2 or iW == 6:
                a[iH][iW] = int(a[iH][iW] * 0.75)
            sum = sum + a[iH][iW]
    cnt = 0.5 * (int(cnt_0) + int(sum / (iMax * iMax)))

    cv2.imwrite(TRAINED_FACES_PATH + "/" + "img/3/" + filename, imgcn)

    # гистограмма
    imgcn = _grab_image(url=url)
    lgh, rec = draw_histogram(imgcn, filename)
    res = int((fcs + cnt + lgh) / 3)

    text = ""
    return 'Изображение обработано ' + "{}{:.2f}".format(text, fm), fcs, cnt, lgh, res, rec
