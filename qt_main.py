import sys
from PyQt6 import QtWidgets, QtCore
from qt_gui import RobotControlWindow
from config import settings


def main():
    app = QtWidgets.QApplication(sys.argv)
    server_proc = QtCore.QProcess()
    server_proc.setProgram(sys.executable)
    server_proc.setArguments(["main.py"])
    server_proc.setWorkingDirectory(".")
    server_proc.start()

    tts_proc = None
    if settings.tts_preload_all and not settings.xtts_use_gpu:
        tts_proc = QtCore.QProcess()
        tts_proc.setProgram("cmd.exe")
        tts_proc.setArguments(["/c", "start", "", "start_tts_services.bat"])
        tts_proc.setWorkingDirectory(".")
        tts_proc.start()
    win = RobotControlWindow()
    win._server_proc = server_proc
    win._tts_proc = tts_proc
    win.show()
    exit_code = app.exec()
    if server_proc.state() != QtCore.QProcess.ProcessState.NotRunning:
        server_proc.terminate()
        server_proc.waitForFinished(3000)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
