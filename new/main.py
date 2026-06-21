"""Entry point: spawns the simulation worker on its own QThread and shows the UI."""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread

from sim_worker import SimulationWorker
from ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    # Build worker first, move to its thread, start thread, then connect UI
    worker = SimulationWorker()
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    window = MainWindow(worker)
    window.show()

    # Clean shutdown: when the app quits, ask the worker to stop and join thread
    def shutdown():
        worker.stop()
        thread.quit()
        thread.wait(2000)
    app.aboutToQuit.connect(shutdown)

    thread.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
