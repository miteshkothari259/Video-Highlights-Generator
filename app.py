import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets

from ui.Ui import Ui_MainWindow

from utils.highlights import highlight_generation
#from utils.download_youtube import youtube
from utils.feature import Feature
from utils.selector import Summarizer

from player import VideoWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.filePath = None
        self.fileName = None
        self.summarization = None
        self.sportsType = None
        self.gif = QtGui.QMovie('utils/icons/gif/progress.gif')
        
        self.setupUi(self)
        self.selectComboBox.setCurrentIndex(-1)
        self.chooseComboBox.setCurrentIndex(-1)
        self.selectComboBox.activated[str].connect(self.onSelect)
        self.chooseComboBox.activated[str].connect(self.onChoose)
        self.browseButton.clicked.connect(self.openFile)
        self.generateButton.clicked.connect(self.generate_summary)
        self.actionExit.triggered.connect(self.exit_application)
        
    def setMovie(self):
        rect = self.outputBox.geometry()
        size = QtCore.QSize(max(rect.width(), rect.height()), max(rect.width(), rect.height()))
        self.gif.setScaledSize(size)
        self.outputBox.setMovie(self.gif)
        
    def onSelect(self,text):
        self.summarization = text.lower()

    def onChoose(self,text):
        self.sportsType = text.lower()
        
    def openFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open VIdeo File', "", "video files (*.mp4)")
        position = fname[0].rfind('/')  + 1
        self.filePath, self.fileName = fname[0][0 : position] , fname[0][position : ]
        self.pathTextBox.setText(fname[0])
        
    def generate_summary(self):
        self.setMovie()
        self.gif.start()
        if self.summarization == 'sports' and self.sportsType == 'youtube link':
            # youtube download
            url = self.urlTextBox.text()
            self.thread = ThreadClass(url = url, do_download = True)
            self.thread.start()
            self.thread.update.connect(self.output)
        else:
            # movie and sports 
            self.thread = ThreadClass(isMovie = self.summarization, filePath = self.filePath, fileName= self.fileName)
            self.thread.start()
            self.thread.update.connect(self.output)

    def resetAll(self):
        self.summarization = None
        self.sportsType = None
        self.gif.stop()
        self.outputBox.clear()
        #self.urlTextBox.clear()
        self.pathTextBox.clear()

    def output(self,text,isError,errorType):
        self.resetAll()
        location, status = text
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        if isError:
            msgBox.setText(status)
            msgBox.setDetailedText(errorType)
            msgBox.setStandardButtons(QMessageBox.Close)
            msgBox.setWindowTitle('Error')

        else:
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setText(status)
            msgBox.setWindowTitle('Success')
        # msgBox.buttonClicked.connect(self.msgButton)
        returnValue = msgBox.exec_()
        if returnValue == QMessageBox.Ok:
            self.player = VideoWindow(fileName=location)
            self.player.show()
            
    def msgButton(self,button):
        print(button)

    def exit_application(self):
        # exiting the application
        sys.exit()
        
class ThreadClass(QtCore.QThread):
    update = QtCore.pyqtSignal(tuple,bool,str)

    def __init__(self,isMovie = False,filePath = None,fileName = None,url = None,do_download = False):
        super(ThreadClass, self).__init__(None)
        self.url = url
        self.do_download = do_download
        self.filePath = filePath
        self.fileName = fileName
        self.isMovie = isMovie

    def run(self):
        # first download youtube video then generate sports highlights
        if self.isMovie != 'movie':
            try:
            # if download is true download the video
                if self.do_download:
                    yt = youtube(self.url)
                    self.filePath, self.fileName= yt.download()
                obj = highlight_generation(self.filePath, self.fileName)
                status = obj.generate()
                self.update.emit(status,False,None)
            except Exception as e:
                self.update.emit(('_','Error occured, Please try again!'),True,str(e))

        else:
            try: 
                f = Feature(self.filePath+self.fileName)
                summarizer = Summarizer(duration=120)
                summarizer.set_feature_extractor(f)
                status = summarizer.summarize()
                self.update.emit(status, False, None)
            except Exception as e: 
                self.update.emit(('_','Error occured, Please try again!'),True,str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
