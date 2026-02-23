# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

mediapipe_datas = collect_data_files("mediapipe")
mediapipe_bins = collect_dynamic_libs("mediapipe")
mediapipe_hidden = collect_submodules("mediapipe")

cv2_datas = collect_data_files("cv2")
cv2_bins = collect_dynamic_libs("cv2")
cv2_hidden = collect_submodules("cv2")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=mediapipe_bins + cv2_bins,
    datas=mediapipe_datas + cv2_datas,
    hiddenimports=mediapipe_hidden + cv2_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gesture_robot_rostech',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)