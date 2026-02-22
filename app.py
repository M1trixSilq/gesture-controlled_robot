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
mp = load_optional_module("mediapipe")


def env_hint_ru() -> str:
    return f"Python: {sys.executable}"


def opencv_missing_message_ru() -> str:
    current_python = Path(sys.executable)
    lines = [
        "Пакет opencv-python недоступен в текущем интерпретаторе.",
        f"Текущий Python: {current_python}",
        "Установите зависимости в этот же интерпретатор:",
        f"  \"{current_python}\" -m pip install -r requirements.txt",
        "И запускайте этим же интерпретатором:",
        f"  \"{current_python}\" main.py",
    ]

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        scripts = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        venv_python = Path(venv) / scripts
        lines.append(f"Обнаружено активное venv: {venv}")
        lines.append(f"Рекомендуемый запуск: \"{venv_python}\" main.py")

    return "\n".join(lines)


class GestureRobotController:
    def __init__(self, camera_index: int = 0, min_detection_conf: float = 0.5, min_tracking_conf: float = 0.5):
        self.camera_index = camera_index
        self.prev_command = COMMANDS["STOP"]
        self.prev_command_time = 0.0
        self.command_hold_sec = 0.25

        self.mp_hands = None
        self.mp_draw = None
        self.hands = None

        if mp is not None:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf,
            )

    @staticmethod
    def _finger_is_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def _is_fist(self, landmarks, is_right_hand: bool) -> bool:
        fingers = [
            self._finger_is_extended(landmarks, 8, 6),
            self._finger_is_extended(landmarks, 12, 10),
            self._finger_is_extended(landmarks, 16, 14),
            self._finger_is_extended(landmarks, 20, 18),
        ]
        thumb_folded = landmarks[4].x < landmarks[3].x if is_right_hand else landmarks[4].x > landmarks[3].x
        return not any(fingers) and thumb_folded

    def _is_open_palm(self, landmarks) -> bool:
        fingers = [
            self._finger_is_extended(landmarks, 8, 6),
            self._finger_is_extended(landmarks, 12, 10),
            self._finger_is_extended(landmarks, 16, 14),
            self._finger_is_extended(landmarks, 20, 18),
        ]
        thumb_extended = abs(landmarks[4].x - landmarks[2].x) > 0.08
        return all(fingers) and thumb_extended

    def _classify_direction(self, landmarks) -> RobotCommand:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y

        horizontal_threshold = 0.08
        vertical_threshold = 0.12

        if dx > horizontal_threshold:
            return COMMANDS["TURN_RIGHT"]
        if dx < -horizontal_threshold:
            return COMMANDS["TURN_LEFT"]
        if dy > vertical_threshold:
            return COMMANDS["BACKWARD"]
        if dy < -vertical_threshold:
            return COMMANDS["FORWARD"]
        return COMMANDS["STOP"]

    def detect_command(self, hand_landmarks, is_right_hand: bool) -> RobotCommand:
        landmarks = hand_landmarks.landmark

        if self._is_fist(landmarks, is_right_hand):
            return COMMANDS["GRIP_CLOSE"]
        direction_command = self._classify_direction(landmarks)
        if direction_command.code != COMMANDS["STOP"].code:
            return direction_command
        if self._is_open_palm(landmarks):
            return COMMANDS["GRIP_OPEN"]
        return COMMANDS["STOP"]

    @staticmethod
    def _draw_hand_box(frame, hand_landmarks):
        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = max(0, min(xs) - 20), min(w - 1, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 20), min(h - 1, max(ys) + 20)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _draw_box_label(frame, box, command: RobotCommand):
        x_min, y_min, _, _ = box
        lines = [f"Жест: {command.gesture_ru}", f"Робот: {command.action_ru}"]
        base_x = x_min + 10
        base_y = max(30, y_min - 35)
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
        if self.hands is None:
            return COMMANDS["STOP"], f"Распознавание жестов недоступно: {env_hint_ru()}"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        rgb.flags.writeable = True

        command = COMMANDS["STOP"]
        status_text = "Кисть не найдена. Робот: Стоп"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            self._draw_hand_box(frame, hand)
            handedness = result.multi_handedness[0].classification[0].label if result.multi_handedness else "Right"
            is_right_hand = handedness == "Right"

            command = self.detect_command(hand, is_right_hand)
            command = self._stabilize_command(command)
            box = self._draw_hand_box(frame, hand)
            self._draw_box_label(frame, box, command)
            status_text = f"Обнаружен жест: {command.gesture_ru}. Команда роботу: {command.action_ru}"

        return command, status_text

    def run(self):
        if cv2 is None:
            raise RuntimeError(opencv_missing_message_ru())

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