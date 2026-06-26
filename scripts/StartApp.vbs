Set WshShell = CreateObject("WScript.Shell")
' Hardcode the absolute path to the project directory so this script works even if moved to the Desktop
projectDir = "C:\Users\Ann\OneDrive\Documents\GitHub\Git Projects\ADAS Version 2"

' Run the batch file without flashing a console window
WshShell.Run chr(34) & projectDir & "\scripts\run.bat" & Chr(34), 1, false
