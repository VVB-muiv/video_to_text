import sys
from PyQt6.QtWidgets import QApplication
from app import VideoDescriptorApp
from processor import Vocabulary

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDescriptorApp()
    window.show()
    sys.exit(app.exec())
