import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QFileDialog, QProgressBar, QMessageBox, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase, QPixmap
from processor import VideoProcessor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl



class VideoDescriptorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        font_id = QFontDatabase.addApplicationFont("D:/учеба_2024/2025/вкр/Dune_Rise.ttf")
        if font_id != -1:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            self.roboto_font = font_families[0] if font_families else "Arial"
        else:
            self.roboto_font = "Arial"  # Fallback шрифт

        self.setWindowTitle("VideoDescriptor")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: #c6e6c6;")

        self.video_path = ""
        self.init_ui()
        # Проверяем наличие файлов модели
        self.model_available = self.check_model_files()
        if not self.model_available:
            self.show_demo_warning()

    def check_model_files(self):
        """Проверка наличия файлов модели"""
        model_path = "resources/final_model.pt"
        vocab_path = "resources/vocabulary.pkl"
        return os.path.exists(model_path) and os.path.exists(vocab_path)

    def show_demo_warning(self):
        """Предупреждение о демо-режиме"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Демо-режим")
        msg.setText("Модель не найдена. Приложение работает в демо-режиме.")
        msg.setInformativeText(
            "Для полной функциональности поместите файлы в папку resources"
        )
        msg.exec()

    def init_ui(self):
        # Создаём центральный виджет и основной макет
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # Горизонтальный layout для заголовка и изображения
        header_layout = QHBoxLayout()

        # Заголовок приложения
        title_label = QLabel("Video Descriptor".upper())
        title_label.setStyleSheet(f"font-family: '{self.roboto_font}'; font-size: 24px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Добавляем в горизонтальный layout
        header_layout.addWidget(title_label)

        main_layout.addLayout(header_layout)
        main_layout.addSpacing(20)

        # Секция выбора файла
        file_layout = QHBoxLayout()
        self.file_path_input = QLabel("Выберите файл")
        self.file_path_input.setStyleSheet("""
            border: 1px solid black;
            background-color: white;
            padding: 5px;
        """)

        self.setup_video_player(main_layout)
        main_layout.addSpacing(20)

        self.browse_button = QPushButton("Обзор")
        self.browse_button.setStyleSheet("""
            background-color: white;
            padding: 5px;
            border: 1px solid black;
        """)
        self.browse_button.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_path_input, 4)
        file_layout.addWidget(self.browse_button, 1)
        main_layout.addLayout(file_layout)
        main_layout.addSpacing(20)

        # Кнопка получения описания
        self.process_button = QPushButton("Получить описание")
        self.process_button.setStyleSheet("""
            background-color: #e6f2e6;
            border: 1px solid black;
            padding: 10px;
            font-weight: bold;
        """)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_video)

        button_container = QHBoxLayout()
        button_container.addStretch()
        button_container.addWidget(self.process_button)
        button_container.addStretch()
        main_layout.addLayout(button_container)
        main_layout.addSpacing(20)

        # Секция результата
        result_label = QLabel("Результат:")
        main_layout.addWidget(result_label)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid black;
                background-color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4ca64c;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        main_layout.addSpacing(10)

        # Область результата
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            background-color: #f2f2e6;
            border: 1px solid black;
        """)
        main_layout.addWidget(self.result_text)

        self.setCentralWidget(central_widget)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "", "Видео файлы (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            file_name = os.path.basename(file_path)
            self.file_path_input.setText(file_name)
            self.process_button.setEnabled(True)

            # Загрузка видео в плеер
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.play_button.setEnabled(True)

    def process_video(self):
        if not self.video_path:
            return

        self.process_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.setText("Обработка видео...")

        # Создаём рабочий поток для обработки видео
        self.worker = VideoProcessorThread(self.video_path)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_result)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_result(self, description):
        self.result_text.setText(description)

    def processing_finished(self):
        self.process_button.setEnabled(True)
        self.browse_button.setEnabled(True)

    def show_error(self, error_message):
        self.result_text.setText(f"Ошибка: {error_message}")
        self.process_button.setEnabled(True)
        self.browse_button.setEnabled(True)

    def setup_video_player(self, main_layout):
        """Настройка видео-плеера с превью"""
        # Создание компонентов медиа-плеера
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Видео-виджет для отображения
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)  # Минимальная высота
        self.video_widget.setStyleSheet("""
            border: 2px solid black;
            background-color: #f0f0f0;
        """)
        self.media_player.setVideoOutput(self.video_widget)

        # Элементы управления
        controls_layout = QHBoxLayout()

        # Кнопка Play/Pause
        self.play_button = QPushButton("▶ Play")
        self.play_button.setEnabled(False)
        self.play_button.setStyleSheet("""
            background-color: #e6f2e6;
            border: 1px solid black;
            padding: 8px 15px;
            font-weight: bold;
        """)
        self.play_button.clicked.connect(self.toggle_playback)

        # Слайдер позиции
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid black;
                height: 8px;
                background: white;
            }
            QSlider::handle:horizontal {
                background: #4ca64c;
                border: 1px solid black;
                width: 18px;
                border-radius: 9px;
            }
        """)

        # Подключение сигналов
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.playbackStateChanged.connect(self.playback_state_changed)

        # Компоновка элементов управления
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.position_slider)

        # Добавление в основной макет
        main_layout.addWidget(self.video_widget)
        main_layout.addLayout(controls_layout)

    def toggle_playback(self):
        """Переключение воспроизведения/паузы"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def set_position(self, position):
        """Установка позиции воспроизведения"""
        self.media_player.setPosition(position)

    def duration_changed(self, duration):
        """Обновление диапазона слайдера при изменении длительности"""
        self.position_slider.setRange(0, duration)

    def position_changed(self, position):
        """Обновление позиции слайдера"""
        self.position_slider.setValue(position)

    def playback_state_changed(self, state):
        """Обновление текста кнопки в зависимости от состояния"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("▶ Play")


class VideoProcessorThread(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            processor = VideoProcessor()

            # Сигнализируем о прогрессе
            self.progress_updated.emit(10)

            # Загрузка модели и словаря
            processor.load_model()
            self.progress_updated.emit(30)

            # Обработка видео
            self.progress_updated.emit(50)
            description = processor.process_video(
                self.video_path,
                progress_callback=self.progress_callback
            )

            # Отправляем результат
            self.result_ready.emit(description)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def progress_callback(self, stage, progress):
        # stage - этап обработки (извлечение кадров, извлечение признаков, генерация)
        # progress - прогресс этапа от 0 до 1

        # Пересчитываем общий прогресс
        if stage == "extract_frames":
            total_progress = 30 + progress * 20  # 30-50%
        elif stage == "extract_features":
            total_progress = 50 + progress * 20  # 50-70%
        elif stage == "generate":
            total_progress = 70 + progress * 30  # 70-100%

        self.progress_updated.emit(int(total_progress))



