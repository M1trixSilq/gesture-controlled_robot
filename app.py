from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path


@dataclass
class RobotCommand:
    code: str
    gesture_ru: str
    action_ru: str


COMMANDS = {
    "STOP": RobotCommand("0000", "Горизонтально", "Стоп"),
    "TURN_RIGHT": RobotCommand("0001", "Наклон вправо", "Поворот направо"),
    "TURN_LEFT": RobotCommand("0010", "Наклон влево", "Поворот налево"),
    "BACKWARD": RobotCommand("1000", "Наклон назад", "Назад"),
    "FORWARD": RobotCommand("0100", "Наклон вперёд", "Вперёд"),
    "GRIP_CLOSE": RobotCommand("GRIP_CLOSE", "Кулак", "Сжать захват манипулятора"),
    "GRIP_OPEN": RobotCommand("GRIP_OPEN", "Открытая кисть", "Разжать захват манипулятора"),
}


def load_optional_module(name: str):
    if not find_spec(name):
        return None
    try:
        return import_module(name)
    except Exception:
        return None


cv2 = load_optional_module("cv2")
np = load_optional_module("numpy")


def env_hint_ru() -> str:
    return f"Python: {sys.executable}"


def opencv_missing_message_ru() -> str:
    current_python = Path(sys.executable)
    lines = [
        "Пакет opencv-python недоступен в текущем интерпретаторе.",
        f"Текущий Python: {current_python}",
        "Установите зависимости в этот же интерпретатор:",
        f'  "{current_python}" -m pip install -r requirements.txt',
        "И запускайте этим же интерпретатором:",
        f'  "{current_python}" main.py',
    ]

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        scripts = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        venv_python = Path(venv) / scripts
        lines.append(f"Обнаружено активное venv: {venv}")
        lines.append(f'Рекомендуемый запуск: "{venv_python}" main.py')

    return "\n".join(lines)


def numpy_missing_message_ru() -> str:
    return (
        "Пакет numpy недоступен в текущем интерпретаторе. "
        "Установите зависимости: python -m pip install -r requirements.txt"
    )


class GestureRobotController:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.prev_command = COMMANDS["STOP"]
        self.prev_command_time = 0.0
        self.command_hold_sec = 0.25

    @staticmethod
    def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5


    def _extract_hand_contour(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 7000:
            return None, mask
        return contour, mask

    def _count_fingers(self, contour, center: tuple[int, int]) -> int:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 4:
            return 0

        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return 0

        fingers = 0
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = self._distance(start, end)
            b = self._distance(start, far)
            c = self._distance(end, far)
            if b * c == 0:
                continue

            angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))
            if angle > 85:
                continue

            if far[1] > center[1]:
                continue

            fingers += 1

        return min(fingers + 1, 5) if fingers > 0 else 0

    def _classify_direction(self, center: tuple[int, int], wrist: tuple[int, int]) -> RobotCommand:
        dx = center[0] - wrist[0]
        dy = center[1] - wrist[1]

        horizontal_threshold = 35
        vertical_threshold = 45

        if dx > horizontal_threshold:
            return COMMANDS["TURN_RIGHT"]
        if dx < -horizontal_threshold:
            return COMMANDS["TURN_LEFT"]
        if dy > vertical_threshold:
            return COMMANDS["BACKWARD"]
        if dy < -vertical_threshold:
            return COMMANDS["FORWARD"]
        return COMMANDS["STOP"]

    def _command_from_shape(self, finger_count: int, center: tuple[int, int], wrist: tuple[int, int]) -> RobotCommand:
        if finger_count <= 1:
            return COMMANDS["GRIP_CLOSE"]
        if finger_count >= 4:
            return COMMANDS["GRIP_OPEN"]

        direction_command = self._classify_direction(center, wrist)
        if direction_command.code != COMMANDS["STOP"].code:
            return direction_command
        return COMMANDS["STOP"]

    @staticmethod
    def _draw_hand_box(frame, contour):
        x, y, w, h = cv2.boundingRect(contour)
        x_min, y_min = max(0, x - 20), max(0, y - 20)
        x_max = min(frame.shape[1] - 1, x + w + 20)
        y_max = min(frame.shape[0] - 1, y + h + 20)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.putText(frame, "Ладонь обнаружена", (x_min, max(25, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _draw_box_label(frame, box, command: RobotCommand, finger_count: int):
        x_min, y_min, _, _ = box
        lines = [f"Жест: {command.gesture_ru}", f"Робот: {command.action_ru}", f"Пальцев: {finger_count}"]
        base_x = x_min + 10
        base_y = max(30, y_min - 55)
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (base_x, base_y + (i * 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

    def _stabilize_command(self, new_command: RobotCommand) -> RobotCommand:
        now = time.time()
        if new_command.code == self.prev_command.code:
            self.prev_command_time = now
            return new_command

        if now - self.prev_command_time < self.command_hold_sec:
            return self.prev_command

        self.prev_command = new_command
        self.prev_command_time = now
        return new_command

    def _process_frame(self, frame) -> tuple[RobotCommand, str]:


        contour, _ = self._extract_hand_contour(frame)
        command = COMMANDS["STOP"]
        status_text = "Кисть не найдена. Робот: Стоп"

        if contour is not None:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                center = (cx, cy)
            else:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

            bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
            finger_count = self._count_fingers(contour, center)
            command = self._command_from_shape(finger_count, center, bottom_point)
            command = self._stabilize_command(command)

            box = self._draw_hand_box(frame, contour)
            self._draw_box_label(frame, box, command, finger_count)
            cv2.circle(frame, center, 8, (255, 0, 255), -1)
            cv2.circle(frame, bottom_point, 8, (0, 255, 255), -1)
            cv2.line(frame, center, bottom_point, (255, 255, 0), 2)

            status_text = f"Обнаружен жест: {command.gesture_ru}. Команда роботу: {command.action_ru}"

        return command, status_text

    def run(self):
        if cv2 is None:
            raise RuntimeError(opencv_missing_message_ru())
        if np is None:
            raise RuntimeError(numpy_missing_message_ru())

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру. Проверьте доступ к устройству.")

        print("Запуск. Нажмите 'q' для выхода.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            command, status_text = self._process_frame(frame)

            cv2.putText(frame, "Управление роботом жестами", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            cv2.putText(frame, f"D3 D2 D1 D0: {command.code}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.imshow("Gesture Robot UI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Управление мобильным роботом с манипулятором по жестам")
    parser.add_argument("--camera", type=int, default=0, help="Индекс камеры (по умолчанию 0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = GestureRobotController(camera_index=args.camera)
    try:
        app.run()
    except RuntimeError as err:
        print(err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())