$ErrorActionPreference = 'Stop'

$fontPath = Join-Path $env:WINDIR 'Fonts/arial.ttf'
if (-not (Test-Path $fontPath)) {
    Write-Error "Font not found: $fontPath. Set a Cyrillic-capable TTF path and rerun."
}

python -m PyInstaller --noconfirm --clean --onefile --name gesture_robot_rostech --add-data "$fontPath;assets/fonts" --hidden-import PIL --hidden-import PIL.Image --hidden-import PIL.ImageDraw --hidden-import PIL.ImageFont main.py

Write-Host 'Build completed. Executable is in dist/gesture_robot_rostech.exe'