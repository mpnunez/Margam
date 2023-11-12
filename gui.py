import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap

def window():
   app = QApplication(sys.argv)
   win = QWidget()
   grid = QGridLayout()
	
   for i in range(0,6):
      for j in range(0,7):
          pixmap = QPixmap('empty.png')
          label = QLabel()
          label.setPixmap(pixmap)
          grid.addWidget(label,i,j)
			
   win.setLayout(grid)
   win.setWindowTitle("PyQt Grid Example")
   win.setGeometry(50,50,200,200)
   win.show()
   sys.exit(app.exec_())

if __name__ == '__main__':
   window()