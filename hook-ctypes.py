from PyInstaller.utils.hooks import collect_dynamic_libs

# Include the _ctypes DLL from the standard library
binaries = collect_dynamic_libs('ctypes')
