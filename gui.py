import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap

def window():
    app = QApplication(sys.argv)
    win = QWidget()
    grid = QGridLayout()
    	
    nrows = 6
    ncols = 7
    for i in range(nrows):
       for j in range(ncols):
           pixmap = QPixmap('empty.png')
           pixmap = pixmap.scaledToWidth(100)
           label = QLabel()
           label.setPixmap(pixmap)
           #label = label.resize(pixmap.width()//2,
           #               pixmap.height()//2)
           grid.addWidget(label,i,j)
           
    for j in range(ncols):
        grid.addWidget(QPushButton("Drop"),nrows,j)
           
    			
    win.setLayout(grid)
    win.setWindowTitle("PyQt Grid Example")
    win.setGeometry(50,50,200,200)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
   window()