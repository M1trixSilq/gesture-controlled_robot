from __future__ import annotations

import argparse
import math
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
    "STOP": RobotCommand("0000", "1 палец вниз", "Стоп"),
    "TURN_RIGHT": RobotCommand("0001", "1 палец вправо", "Поворот направо"),
    "TURN_LEFT": RobotCommand("0010", "1 палец влево", "Поворот налево"),
    "BACKWARD": RobotCommand("1000", "2 пальца вверх", "Назад"),
    "FORWARD": RobotCommand("0100", "1 палец вверх", "Вперёд"),
    "GRIP_CLOSE": RobotCommand("GRIP_CLOSE", "Кулак", "Сжать захват манипулятора"),
    "GRIP_OPEN": RobotCommand("GRIP_OPEN", "Открытая ладонь", "Разжать захват манипулятора"),
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
    def _distance(lm, a: int, b: int) -> float:
        dx = lm[a].x - lm[b].x
        dy = lm[a].y - lm[b].y
        dz = lm[a].z - lm[b].z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @staticmethod
    def _joint_angle(lm, a: int, b: int, c: int) -> float:
        ab = (lm[a].x - lm[b].x, lm[a].y - lm[b].y, lm[a].z - lm[b].z)
        cb = (lm[c].x - lm[b].x, lm[c].y - lm[b].y, lm[c].z - lm[b].z)
        dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
        ab_norm = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
        cb_norm = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
        if ab_norm < 1e-6 or cb_norm < 1e-6:
            return 0.0
        cos_val = max(-1.0, min(1.0, dot / (ab_norm * cb_norm)))
        return math.degrees(math.acos(cos_val))

    @staticmethod
    def _thumb_extended(hand_landmarks) -> bool:
        lm = hand_landmarks.landmark
        thumb_straight = GestureRobotController._joint_angle(lm, 2, 3, 4) > 155
        thumb_far_from_palm = GestureRobotController._distance(lm, 4, 5) > GestureRobotController._distance(lm, 3, 5) * 1.15
        thumb_far_from_wrist = GestureRobotController._distance(lm, 4, 0) > GestureRobotController._distance(lm, 3, 0) * 1.08
        return thumb_straight and thumb_far_from_palm and thumb_far_from_wrist


    @staticmethod
    def _fingers_up(hand_landmarks, handedness_label: str) -> list[bool]:
        lm = hand_landmarks.landmark
        thumb_up = GestureRobotController._thumb_extended(hand_landmarks)

        def is_finger_extended(mcp: int, pip: int, dip: int, tip: int) -> bool:
            pip_angle = GestureRobotController._joint_angle(lm, mcp, pip, tip)
            dip_angle = GestureRobotController._joint_angle(lm, pip, dip, tip)
            tip_farther_than_pip = GestureRobotController._distance(lm, tip, 0) > GestureRobotController._distance(lm, pip, 0) * 1.08
            return pip_angle > 160 and dip_angle > 150 and tip_farther_than_pip

        others = [
            is_finger_extended(5, 6, 7, 8),
            is_finger_extended(9, 10, 11, 12),
            is_finger_extended(13, 14, 15, 16),
            is_finger_extended(17, 18, 19, 20),
        ]
        return [thumb_up, *others]

    @staticmethod
    def _is_fist(hand_landmarks, fingers: list[bool]) -> bool:
        lm = hand_landmarks.landmark
        curled_others = [
            GestureRobotController._joint_angle(lm, 5, 6, 8) < 115,
            GestureRobotController._joint_angle(lm, 9, 10, 12) < 115,
            GestureRobotController._joint_angle(lm, 13, 14, 16) < 115,
            GestureRobotController._joint_angle(lm, 17, 18, 20) < 115,
        ]
        thumb_curled = not GestureRobotController._thumb_extended(hand_landmarks)
        return all(curled_others) and thumb_curled

    @staticmethod
    def _open_palm_geometry_metrics(hand_landmarks) -> tuple[float, float, float, bool]:
        lm = hand_landmarks.landmark

        def dist(a: int, b: int) -> float:
            return GestureRobotController._distance(lm, a, b)

        palm_size = max(dist(0, 9), 1e-6)

        adjacent_tip_distances = [dist(8, 12), dist(12, 16), dist(16, 20)]
        avg_tip_norm = sum(adjacent_tip_distances) / (len(adjacent_tip_distances) * palm_size)

        # Инвариант к масштабу: расстояние между tip относительно расстояния между MCP.
        tip_to_base_ratios = [
            dist(8, 12) / max(dist(5, 9), 1e-6),
            dist(12, 16) / max(dist(9, 13), 1e-6),
            dist(16, 20) / max(dist(13, 17), 1e-6),
        ]
        avg_tip_to_base_ratio = sum(tip_to_base_ratios) / len(tip_to_base_ratios)

        thumb_index_norm = dist(4, 8) / palm_size

        fingers_vertical = all(
            (lm[tip].y - lm[mcp].y) / palm_size < -0.45
            for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))
        )

        return avg_tip_norm, avg_tip_to_base_ratio, thumb_index_norm, fingers_vertical

    @staticmethod
    def _is_open_palm_spread(hand_landmarks, fingers: list[bool]) -> bool:
        # Большой палец может флапать в детекции, поэтому опираемся на 4 длинных пальца.
        if not all(fingers[1:]):
            return False

        avg_tip_norm, avg_tip_to_base_ratio, thumb_index_norm, fingers_vertical = (
            GestureRobotController._open_palm_geometry_metrics(hand_landmarks)
        )

        thumb_wide_or_up = fingers[0] or thumb_index_norm > 0.90
        return (
            fingers_vertical
            and avg_tip_norm > 0.62
            and avg_tip_to_base_ratio > 1.45
            and thumb_wide_or_up
        )

    @staticmethod
    def _is_open_palm_together(hand_landmarks, fingers: list[bool]) -> bool:
        if not all(fingers[1:]):
            return False

        avg_tip_norm, avg_tip_to_base_ratio, thumb_index_norm, fingers_vertical = (
            GestureRobotController._open_palm_geometry_metrics(hand_landmarks)
        )

        thumb_not_spread = (not fingers[0]) or thumb_index_norm < 0.90
        return (
            fingers_vertical
            and avg_tip_norm < 0.58
            and avg_tip_to_base_ratio < 1.35
            and thumb_not_spread
        )

    @staticmethod
    def _classify_single_finger_direction(hand_landmarks, fingers: list[bool]) -> RobotCommand:
        lm = hand_landmarks.landmark
        horizontal_threshold = 0.28
        vertical_threshold = 0.55

        finger_vectors = {
            0: (2, 4),
            1: (5, 8),
            2: (9, 12),
            3: (13, 16),
            4: (17, 20),
        }

        finger_idx = next((idx for idx, up in enumerate(fingers) if up and idx != 0), None)
        if finger_idx is None:
            return COMMANDS["STOP"]

        base_id, tip_id = finger_vectors[finger_idx]
        dx = lm[tip_id].x - lm[base_id].x
        dy = lm[tip_id].y - lm[base_id].y

        palm_scale = max(GestureRobotController._distance(lm, 0, 9), 1e-6)
        dx /= palm_scale
        dy /= palm_scale
        total = abs(dx) + abs(dy)
        if total < 0.35:
            return COMMANDS["STOP"]

        horizontal_share = abs(dx) / total
        vertical_share = abs(dy) / total

        if dx > horizontal_threshold and horizontal_share > 0.60:
            return COMMANDS["TURN_RIGHT"]
        if dx < -horizontal_threshold and horizontal_share > 0.60:
            return COMMANDS["TURN_LEFT"]   
        if dy < -vertical_threshold and vertical_share > 0.60:
            return COMMANDS["FORWARD"]
        if dy > vertical_threshold and vertical_share > 0.60:
            return COMMANDS["STOP"]
        return COMMANDS["STOP"]

    @staticmethod
    def _is_two_fingers_up_backward(hand_landmarks, fingers: list[bool]) -> bool:
        # Назад: подняты только указательный и средний пальцы, оба направлены вверх.
        if not (fingers[1] and fingers[2] and not fingers[3] and not fingers[4]):
            return False

        lm = hand_landmarks.landmark
        index_dy = lm[8].y - lm[5].y
        middle_dy = lm[12].y - lm[9].y
        palm_scale = max(GestureRobotController._distance(lm, 0, 9), 1e-6)
        vertical_threshold = 0.55
        fingers_parallel = abs((lm[8].x - lm[5].x) - (lm[12].x - lm[9].x)) / palm_scale < 0.35
        return (index_dy / palm_scale) < -vertical_threshold and (middle_dy / palm_scale) < -vertical_threshold and fingers_parallel

    def _command_from_landmarks(self, hand_landmarks, handedness_label: str) -> tuple[RobotCommand, int]:
        fingers = self._fingers_up(hand_landmarks, handedness_label)
        finger_count = sum(fingers)

        only_index = fingers[1] and not any(fingers[2:])
        only_middle = fingers[2] and not fingers[1] and not fingers[3] and not fingers[4]
        only_ring = fingers[3] and not fingers[1] and not fingers[2] and not fingers[4]
        only_pinky = fingers[4] and not any(fingers[1:4])

        if only_index or only_middle or only_ring or only_pinky:
            return self._classify_single_finger_direction(hand_landmarks, fingers), finger_count

        if self._is_fist(hand_landmarks, fingers):
            return COMMANDS["GRIP_CLOSE"], finger_count

        if self._is_two_fingers_up_backward(hand_landmarks, fingers):
            return COMMANDS["BACKWARD"], finger_count

        if self._is_open_palm_spread(hand_landmarks, fingers) or self._is_open_palm_together(hand_landmarks, fingers):
            return COMMANDS["GRIP_OPEN"], finger_count


        return COMMANDS["STOP"], finger_count

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