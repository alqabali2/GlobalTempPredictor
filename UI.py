import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import arabic_reshaper
from bidi.algorithm import get_display
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QComboBox, QLabel
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ✅ ضبط الخط لدعم العربية في الرسومات فقط
plt.rcParams['font.family'] = 'Arial'

# ✅ تحويل النصوص العربية لإصلاح ترتيب الأحرف فقط في الرسومات البيانية
def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# 🔹 تحميل بيانات درجات الحرارة
df = pd.read_csv(r"C:\Users\Adel\Desktop\TemperaturesByCountry\cleaned_data.csv", parse_dates=["dt"])

class TemperatureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ✅ تصميم حديث لعام 2025
        self.setWindowTitle("تحليل وتوقع درجات الحرارة")
        self.setGeometry(100, 100, 1000, 650)  # توسيع النافذة للحصول على تجربة أفضل
        self.center_window()  # توسيط النافذة على الشاشة

        # 🎨 ضبط تصميم الألوان الحديثة
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;  /* لون خلفية عصري */
            }
            QLabel {
                color: #FFFFFF;  /* اللون الأبيض للنصوص */
                font-size: 16px;
                font-weight: bold;
            }
            QComboBox, QPushButton {
                background-color: #3A3A5A; 
                color: #FFFFFF;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4C4C7F;
            }
            QComboBox::drop-down {
                background-color: #4C4C7F;
            }
        """)

        # ✅ الواجهة الرئيسية
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # ✅ قائمة الدول
        self.country_label = QLabel("اختر الدولة:")
        self.layout.addWidget(self.country_label)
        self.country_selector = QComboBox()
        self.layout.addWidget(self.country_selector)

        # ✅ زر عرض البيانات
        self.load_button = QPushButton("📊 عرض البيانات")
        self.layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.plot_temperature)

        # ✅ زر توقع البيانات
        self.forecast_button = QPushButton("🔮 توقع درجات الحرارة")
        self.layout.addWidget(self.forecast_button)
        self.forecast_button.clicked.connect(self.forecast_temperature)

        # ✅ مخطط البيانات
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # ✅ تحميل قائمة الدول
        self.load_countries()

    def center_window(self):
        """ توسيط النافذة في منتصف الشاشة """
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.geometry()
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        self.move(x, y)

    def load_countries(self):
        """تحميل قائمة الدول من البيانات."""
        countries = sorted(df["Country"].unique())
        self.country_selector.addItems(countries)

    def plot_temperature(self):
        """عرض المخطط الزمني لدرجات الحرارة للدولة المختارة."""
        country = self.country_selector.currentText()
        country_data = df[df["Country"] == country].set_index("dt")["AverageTemperature"].dropna()

        self.ax.clear()
        self.ax.plot(country_data, label=fix_arabic_text(f"درجات الحرارة في {country}"), color="#4CAF50", linewidth=1.8)
        self.ax.set_title(fix_arabic_text(f"التغير في درجات الحرارة - {country}"), fontsize=14, fontweight="bold", color="white")
        self.ax.set_xlabel(fix_arabic_text("السنة"), fontsize=12, fontweight="bold", color="white")
        self.ax.set_ylabel(fix_arabic_text("متوسط درجة الحرارة (°C)"), fontsize=12, fontweight="bold", color="white")
        self.ax.legend(facecolor="#3A3A5A", edgecolor="white")
        self.ax.grid(True, linestyle="--", alpha=0.5, color="gray")
        self.canvas.draw()

    def forecast_temperature(self):
        """تنفيذ توقعات ARIMA و SARIMA وعرضها حتى عام 2040."""
        country = self.country_selector.currentText()
        country_data = df[df["Country"] == country].set_index("dt")["AverageTemperature"].dropna()

        if country_data.empty:
            return

        # حساب عدد الأشهر حتى 2040
        last_date = country_data.index[-1]
        forecast_steps = (2040 - last_date.year) * 12 + (12 - last_date.month)

        # تنفيذ نموذج ARIMA
        arima_model = ARIMA(country_data, order=(1, 1, 1)).fit()
        arima_forecast = arima_model.forecast(steps=forecast_steps)

        # تنفيذ نموذج SARIMA
        sarima_model = SARIMAX(country_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        sarima_forecast = sarima_model.forecast(steps=forecast_steps)

        forecast_index = pd.date_range(start=last_date, periods=forecast_steps, freq="MS")

        self.ax.clear()
        self.ax.plot(country_data, label=fix_arabic_text(f"البيانات الفعلية - {country}"), color="#4CAF50", linewidth=1.8)
        self.ax.plot(forecast_index, arima_forecast, label=fix_arabic_text("توقع ARIMA"), color="#FF9800", linestyle="dashed", linewidth=1.8)
        self.ax.plot(forecast_index, sarima_forecast, label=fix_arabic_text("توقع SARIMA"), color="#03A9F4", linestyle="dashed", linewidth=1.8)
        self.ax.set_title(fix_arabic_text(f"التوقعات المستقبلية لدرجات الحرارة - {country}"), fontsize=14, fontweight="bold", color="white")
        self.ax.set_xlabel(fix_arabic_text("السنة"), fontsize=12, fontweight="bold", color="white")
        self.ax.set_ylabel(fix_arabic_text("متوسط درجة الحرارة (°C)"), fontsize=12, fontweight="bold", color="white")
        self.ax.legend(facecolor="#3A3A5A", edgecolor="white")
        self.ax.grid(True, linestyle="--", alpha=0.5, color="gray")
        self.canvas.draw()

# ✅ تشغيل التطبيق
def run_app():
    app = QApplication(sys.argv)
    window = TemperatureApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
