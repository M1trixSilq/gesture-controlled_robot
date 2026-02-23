@echo off
setlocal

set "FONT_PATH=%WINDIR%\Fonts\arial.ttf"
if not exist "%FONT_PATH%" (
  echo Font not found: %FONT_PATH%
  echo Set FONT_PATH to a Cyrillic-capable TTF and rerun.
  exit /b 1
)

python -m PyInstaller --noconfirm --clean --onefile --name gesture_robot_rostech ^
  --collect-all mediapipe ^
  --collect-all cv2 ^
  --hidden-import numpy ^
  --hidden-import PIL ^
  --hidden-import PIL.Image ^
  --hidden-import PIL.ImageDraw ^
  --hidden-import PIL.ImageFont ^
  --add-data "%FONT_PATH%;assets/fonts" ^
  main.py
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo Build completed. Executable is in dist\gesture_robot_rostech.exe
endlocal