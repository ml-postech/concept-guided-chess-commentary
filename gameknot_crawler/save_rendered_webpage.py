import sys  
import os
import argparse
from lxml import html 
import pickle
import time
import functools

from PyQt5.QtGui import *  
from PyQt5.QtCore import *  
#from PyQt5.QtWebKit import *  
# from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PyQt5 import QtGui, QtCore
# from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
#from PyQt5.QtWebKitWidgets import QWebView

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-software-rasterizer --disable-gpu-compositing --disable-gl-drawing-for-tests"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def parseArguments():
	parser = argparse.ArgumentParser()
	#parser.add_argument("-typ", dest="typ", help="home or subsequent", default='home')
	parser.add_argument("-i", type=int, dest="i", help="i")
	parser.add_argument("-num", type=int, dest="num", help="num")
	args = parser.parse_args()  
	return args
params = parseArguments()
#typ = params.typ

class MainWindow(QMainWindow):
	def __init__(self, url, i, num):
		super(MainWindow, self).__init__()
		self.browser = QWebEngineView()
		self.browser.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, False)
		self.browser.load(QUrl(url))
		self.browser.loadFinished.connect(self.test)
		self.setCentralWidget(self.browser)
		self.showMaximized()
		self.i = i
		self.num = num

	def test(self):
		print('super')
		frame = self.browser.page()
		frame.toHtml(self.callback)

	def callback(self, html):
		i = self.i	
		num = self.num
		# print(unicode(html).encode('utf-8'))
		# html_doc = html.toAscii()
		html_doc = html
		print('to ascii')	
		if num==0:
			fw = open("./saved_files/html/saved"+str(i)+".html", "w")
		else:
			fw = open("./saved_files/html/saved"+str(i)+"_" + str(num) + ".html", "w")
		fw.write(html_doc)
		fw.close()
		self.close()



def save_all():
	global cur_url
	global html_doc
	all_links = pickle.load( open("./saved_files/saved_links.p", "rb") )
	#extra_links = pickle.load( open("extra_pages.p", "r") )
	print("len(all_links) = ",len(all_links))
	num = sys.argv[1]

	i = params.i
	print("i = ",type(i))
	num = params.num
	url = all_links[i]
	if num!=0:
		url+="&pg="+str(num)
	print("i, url = ",i,url)
	#This step is important.Converting QString to Ascii for lxml to process
	#archive_links = html.fromstring(str(result.toAscii()))

	cur_url = url
	error_count = 0
	#try:
	app = QApplication(sys.argv)
	QApplication.setApplicationName('crawler')
	window = MainWindow(cur_url, i, num)
	app.exec_()

if __name__=="__main__":
	save_all()	

'''
s = "https://gameknot.com/annotation.pl/fierce-queen-taking-spanish-easy?gm=63368"

url = 'http://pycoders.com/archive/'  
url = s
r = Render(url)  
result = r.frame.toHtml()
#This step is important.Converting QString to Ascii for lxml to process
archive_links = html.fromstring(str(result.toAscii()))
print archive_links
print "---------"
print result.toAscii()
print "======================================"
print result

'''
