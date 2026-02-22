from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path


def load_optional_module(name: str):
    if not find_spec(name):
        return None
    try:
        return import_module(name)
    except Exception:
        return None


cv2 = load_optional_module("cv2")
np = load_optional_module("numpy")
mp = load_optional_module("mediapipe")
PIL_image = load_optional_module("PIL.Image")
PIL_draw = load_optional_module("PIL.ImageDraw")
PIL_font = load_optional_module("PIL.ImageFont")


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


def mediapipe_missing_message_ru() -> str:
    return (
        "Пакет mediapipe недоступен в текущем интерпретаторе. "
        "Установите зависимости: python -m pip install -r requirements.txt"
    )


def mediapipe_assets_missing_message_ru(err: Exception) -> str:
    return (
        "MediaPipe не смог загрузить внутренние ресурсы (binarypb). "
        "Частая причина — кириллица в пути проекта/venv на Windows.\n"
        "Решение: перенесите проект и виртуальное окружение в путь только с латиницей, "
        "переустановите зависимости и запустите снова.\n"
        f"Детали: {err}"
    )


class GestureRobotController:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.prev_command = COMMANDS["STOP"]
        self.prev_command_time = 0.0
        self.command_hold_sec = 0.25
        self.text_font = self._load_font()

    @staticmethod
    def _load_font():
        if PIL_font is None:
            return None

        candidates = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for font_path in candidates:
            try:
                return PIL_font.truetype(font_path, 28)
            except Exception:
                continue
        return None

    def _draw_text_ru(self, frame, text: str, org: tuple[int, int], color=(255, 255, 255), scale: float = 0.7):
        if PIL_image is None or PIL_draw is None or self.text_font is None:
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL_image.fromarray(rgb)
        draw = PIL_draw.Draw(pil_img)
        draw.text(org, text, font=self.text_font, fill=(int(color[2]), int(color[1]), int(color[0])))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fingers_up(hand_landmarks, handedness_label: str) -> list[bool]:
        lm = hand_landmarks.landmark

        thumb_tip = lm[4]
        thumb_ip = lm[3]
        thumb_up = thumb_tip.x < thumb_ip.x if handedness_label == "Right" else thumb_tip.x > thumb_ip.x

        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        others = [lm[tip].y < lm[pip].y for tip, pip in finger_pairs]
        return [thumb_up, *others]


    @staticmethod
    def _classify_direction(wrist, middle_mcp) -> RobotCommand:
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y


        horizontal_threshold = 0.08
        vertical_threshold = 0.10

        if dx > horizontal_threshold:
            return COMMANDS["TURN_RIGHT"]
        if dx < -horizontal_threshold:
            return COMMANDS["TURN_LEFT"]
        if dy > vertical_threshold:
            return COMMANDS["BACKWARD"]
        if dy < -vertical_threshold:
            return COMMANDS["FORWARD"]
        return COMMANDS["STOP"]

    def _command_from_landmarks(self, hand_landmarks, handedness_label: str) -> tuple[RobotCommand, int]:
        fingers = self._fingers_up(hand_landmarks, handedness_label)
        finger_count = sum(fingers)

        if finger_count <= 1:
            return COMMANDS["GRIP_CLOSE"], finger_count
        if finger_count >= 4:
            return COMMANDS["GRIP_OPEN"], finger_count


        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        return self._classify_direction(wrist, middle_mcp), finger_count

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

    def _draw_box_and_labels(self, frame, hand_landmarks, command: RobotCommand, finger_count: int):
        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = max(0, min(xs) - 20), min(w - 1, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 20), min(h - 1, max(ys) + 20)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        frame = self._draw_text_ru(frame, "Рука обнаружена", (x_min, max(10, y_min - 35)), (0, 255, 0))

        lines = [
            f"Жест: {command.gesture_ru}",
            f"Робот: {command.action_ru}",
            f"Пальцев: {finger_count}",
        ]
        for i, line in enumerate(lines):
            frame = self._draw_text_ru(frame, line, (x_min + 8, min(h - 30, y_max + 10 + i * 30)))
        return frame

    def run(self):
        if cv2 is None:
            raise RuntimeError(opencv_missing_message_ru())
        if np is None:
            raise RuntimeError(numpy_missing_message_ru())
        if mp is None:
            raise RuntimeError(mediapipe_missing_message_ru())

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру. Проверьте доступ к устройству.")

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils


        print("Запуск. Нажмите 'q' для выхода.")
        try:
            hands_ctx = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        except FileNotFoundError as err:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError(mediapipe_assets_missing_message_ru(err)) from err

        with hands_ctx as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                command = COMMANDS["STOP"]
                status_text = "Рука не найдена. Робот: Стоп"

                if result.multi_hand_landmarks and result.multi_handedness:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    handedness = result.multi_handedness[0].classification[0].label
                    command, finger_count = self._command_from_landmarks(hand_landmarks, handedness)
                    command = self._stabilize_command(command)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    frame = self._draw_box_and_labels(frame, hand_landmarks, command, finger_count)
                    status_text = f"Обнаружен жест: {command.gesture_ru}. Команда роботу: {command.action_ru}"

                frame = self._draw_text_ru(frame, "Управление роботом жестами", (20, 20), (0, 255, 0))
                frame = self._draw_text_ru(frame, "Режим: MediaPipe", (20, 55), (0, 220, 220))
                frame = self._draw_text_ru(frame, status_text, (20, 90))
                frame = self._draw_text_ru(frame, f"D3 D2 D1 D0: {command.code}", (20, 125), (255, 255, 0))

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