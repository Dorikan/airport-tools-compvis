from ultralytics.utils.plotting import Annotator, colors
from utils.config import Config


async def draw_results(image, results: dict):
    """
    Рисует результаты детекции средствами ultralytics.

    :param image: изображение
    :param results: результат работы ансамбля
    """

    annotator = Annotator(image, example=str(Config.CLASSES_DICT))

    # Преобразуем боксы в абсолютные координаты
    for r in results:
        box = r['bbox']
        cls = r['class']
        conf = r['confidence']

        if conf < 0.7:
            continue
        x1, y1, x2, y2 = box
        label = f"{Config.CLASSES_DICT[int(cls)]} {conf:.2f}"
        annotator.box_label((x1, y1, x2, y2), label, color=colors(int(cls), True))

    im = annotator.result()
    return im
