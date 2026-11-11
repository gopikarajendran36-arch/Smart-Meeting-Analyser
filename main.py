from PySide6.QtWidgets import QFileDialog, QMessageBox
import sys
import os

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import Qt


class CommitteeMeetingAdvisor(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Meeting Analyser")
        self.setMinimumSize(1100, 700)
        self.resize(1250, 780)

        self.create_dashboard()

    def create_dashboard(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
    
        # ---------------- SIDEBAR ----------------

        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #172554;
                border: none;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 20)
        sidebar_layout.setSpacing(15)

        app_title = QLabel("SMA")
        app_title.setStyleSheet("""
            color: white;
            font-size: 28px;
            font-weight: bold;
            padding: 10px;
        """)

        app_subtitle = QLabel("Smart Meeting\nAnalyser")
        app_subtitle.setStyleSheet("""
            color: #BFDBFE;
            font-size: 14px;
            padding-left: 10px;
        """)

        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(app_subtitle)
        sidebar_layout.addSpacing(30)

        menu_items = [
            "Dashboard",
            "New Meeting",
            "Transcript",
            "Summary",
            "Action Items",
            "Meeting History"
        ]

        for item in menu_items:
            button = QPushButton(item)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(45)

            button.setStyleSheet("""
                QPushButton {
                    color: white;
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                    text-align: left;
                    padding-left: 15px;
                    font-size: 14px;
                }

                QPushButton:hover {
                    background-color: #1E40AF;
                }
            """)

            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch()

        version = QLabel("Version 1.0")
        version.setStyleSheet("color: #93C5FD; padding: 10px;")
        sidebar_layout.addWidget(version)

        # ---------------- MAIN CONTENT ----------------

        content = QWidget()
        content.setStyleSheet("background-color: #F1F5F9;")

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(35, 30, 35, 30)
        content_layout.setSpacing(25)

        # Header

        header_layout = QHBoxLayout()

        welcome = QLabel("Welcome to your Dashboard")
        welcome.setStyleSheet("""
            font-size: 25px;
            font-weight: bold;
            color: #172554;
        """)

        new_meeting_button = QPushButton("+ New Meeting")
        new_meeting_button.setFixedSize(150, 42)
        new_meeting_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        new_meeting_button.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        new_meeting_button.clicked.connect(self.select_meeting_video)

        header_layout.addWidget(welcome)
        header_layout.addStretch()
        header_layout.addWidget(new_meeting_button)

        content_layout.addLayout(header_layout)

        subtitle = QLabel(
            "Manage your committee meetings with AI-powered insights."
        )
        subtitle.setStyleSheet("""
            color: #64748B;
            font-size: 14px;
        """)

        content_layout.addWidget(subtitle)

        # ---------------- DASHBOARD CARDS ----------------

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        card_data = [
            ("Total Meetings", "0", "Meetings processed"),
            ("Transcriptions", "0", "Transcripts generated"),
            ("Action Items", "0", "Tasks identified")
        ]

        for title, number, description in card_data:

            card = QFrame()
            card.setMinimumHeight(140)
            card.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 12px;
                    border: 1px solid #E2E8F0;
                }
            """)

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 18, 20, 18)

            card_title = QLabel(title)
            card_title.setStyleSheet("""
                color: #64748B;
                font-size: 14px;
                border: none;
            """)

            card_number = QLabel(number)
            card_number.setStyleSheet("""
                color: #2563EB;
                font-size: 30px;
                font-weight: bold;
                border: none;
            """)

            card_description = QLabel(description)
            card_description.setStyleSheet("""
                color: #94A3B8;
                font-size: 12px;
                border: none;
            """)

            card_layout.addWidget(card_title)
            card_layout.addWidget(card_number)
            card_layout.addWidget(card_description)

            cards_layout.addWidget(card)

        content_layout.addLayout(cards_layout)

        # ---------------- RECENT MEETINGS ----------------

        recent_title = QLabel("Recent Meetings")
        recent_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #172554;
        """)

        content_layout.addWidget(recent_title)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels([
            "Meeting Name",
            "Date",
            "Status",
            "Action"
        ])

        table.setRowCount(0)

        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(250)

        table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                gridline-color: #E2E8F0;
                color: #334155;
                font-size: 13px;
            }

            QHeaderView::section {
                background-color: #DBEAFE;
                color: #172554;
                padding: 12px;
                border: none;
                font-weight: bold;
            }
        """)

        content_layout.addWidget(table)
        content_layout.addStretch()

        # Add sidebar and content to main layout

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content)

        central_widget.setLayout(main_layout)

    def select_meeting_video(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Meeting Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if file_path:
            self.selected_video_path = file_path
            self.show_processing_screen()

    def show_processing_screen(self):

        dialog = QDialog(self)
        dialog.setWindowTitle("Meeting Processing")
        dialog.setMinimumSize(650, 450)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #F1F5F9;
            }

            QLabel {
                color: #172554;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Heading

        heading = QLabel("Meeting Processing")
        heading.setStyleSheet("""
            font-size: 25px;
            font-weight: bold;
            color: #1D4ED8;
        """)

        layout.addWidget(heading)

        subtitle = QLabel(
            "Your meeting video is ready for AI analysis."
        )
        subtitle.setStyleSheet("font-size: 14px; color: #64748B;")

        layout.addWidget(subtitle)

        # Video details

        video_name = os.path.basename(self.selected_video_path)
        video_size = os.path.getsize(
            self.selected_video_path
        ) / (1024 * 1024)

        details = QLabel(
            f"Video Name: {video_name}\n\n"
            f"File Size: {video_size:.2f} MB\n\n"
            f"Location: {self.selected_video_path}"
        )

        details.setWordWrap(True)
        details.setStyleSheet("""
            background-color: white;
            border: 1px solid #DBEAFE;
            border-radius: 10px;
            padding: 20px;
            font-size: 14px;
        """)

        layout.addWidget(details)

        # Progress bar

        progress = QProgressBar()
        progress.setValue(0)

        progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                height: 20px;
                text-align: center;
                background-color: white;
            }

            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 7px;
            }
        """)

        layout.addWidget(progress)

        # Status

        status = QLabel("Status: Ready to process")
        status.setStyleSheet("""
            font-size: 14px;
            color: #2563EB;
            font-weight: bold;
        """)

        layout.addWidget(status)

        # Start button

        start_button = QPushButton("Start Processing")
        start_button.setFixedHeight(45)

        start_button.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)

        start_button.clicked.connect(
            lambda: status.setText(
                "Backend integration is our next step."
            )
        )

        layout.addWidget(start_button)

        dialog.exec()
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CommitteeMeetingAdvisor()
    window.show()

    sys.exit(app.exec())
