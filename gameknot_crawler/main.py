import json
import numpy as np
import os
import unicodedata
import pickle
import sys
import time
from tqdm import tqdm 
import traceback

from utilities import Utilities

mode = sys.argv[1] # {expand, html_parser}

##############################################################

'''
- Input:
	- saved_links.p
- Go through the links
- Parse the html to see parginator
- Output
	- extra_pages.p : count of number of pages - in same order as in saved_links.p
'''
def expander_main():
	utils = Utilities()
	data_path = "./saved_files/"
	src = data_path + "saved_links.p"
	destination_path = data_path + "expanded_links.p"
	saved_links = pickle.load( open(src,"rb") )
	counts = []
	for i, link in enumerate(saved_links):
		print("i,link = ",i, link)
		num = 0
		try:
			file = "saved"+str(i)+".html"
			html_doc = open(data_path + file,"rb").read()
			soup = utils.getSoupFromHTML(html_doc)
			paginator = utils.getTableOfClass(soup, 'paginator')
			print("len(paginator) = ",len(paginator))
			if len(paginator)>0:
				paginator = paginator[0] # If it occurs, it occurs twice - and both seem to be identical
				txt = utils.soupToText(paginator).lower()
				if txt.count("next")>0:
					txt = txt.replace("pages:","").strip()
					for j,c in enumerate(txt):
						if c>='1' and c<='9': # assuming <=9 extra pages
							num+=1
						else:
							break
			print("num = ",num)
		except:
			print("----ERROR:")
		counts.append(num)
	print("len(counts) = ",len(counts))
	print("sum(counts) = ",sum(counts))
	pickle.dump(counts, open("extra_pages.p","w"))
			


			


##############################################################
class DataCollector:

	def __init__(self):
		self._utils = Utilities()
		self._data_path = "./saved_files/html/"
		self._destination_path = "./outputs/"
		print("------")

	def _getList(self):
		files = os.listdir(self._data_path)
		files = [f for f in files if f.count("html")>0]
		# return ["saved8391.html"] # debug mode
		return files

	def _getBoardValues(self, soup):
		divs = soup.find_all("div", {"data-chess-diagram": True}, recursive=True)
		div = divs[0]
		fen = div["data-chess-diagram"]
		return fen

	def getData(self):
		all_files = self._getList()
		fw = open("error_files.txt","w")
		for file in tqdm(all_files):
			try:
				print("file = ",file)
				html_doc = open(self._data_path + file,"rb").read()
				soup = self._utils.getSoupFromHTML(html_doc)
				results = soup.find_all("table",{"class":"dialog"})
				tmp = results[0] # expecting only 1 table of this type
				
				print(tmp)
				results2 = tmp.find_all("tr")
				print(results2)
				results3 = [ result for result in results2 if len(result.find_all("td", recursive=False))==2 and len(soup.find_all("div", {"data-chess-diagram": True}, recursive=True)) > 0 ] #based on observation
				print(results3)
				all_steps_info = []
				for index, result in enumerate(results3):
					if index % 2==1:
						continue 
					#print "result = ",result
					td_res = result.find_all("td", recursive=False)
					td = td_res[0] ## move+board
					##--- Extract moves
					# print('td:', td)
					txt =  td.get_text()
					moves = txt[:txt.find("<!--")].strip()
					##--- Extract board elements
					fen_with_last = self._getBoardValues(result)
					td = td_res[1] ## Comment
					comment = self._utils.soupToText(td)
					##--- Add to data structure
					current_step_info = [moves, fen_with_last, comment]
					print('==========MOVES==========')
					print(moves)
					print('==========BOARD==========')
					print(fen_with_last)
					print('==========COMMENT==========')
					print(comment)
					all_steps_info.append(current_step_info)
				pickle.dump( all_steps_info, open(self._destination_path + file.replace(".html",".obj"), "wb") )
			except Exception as e:
				print(f"error {e}")
				traceback.print_exc()
				fw.write(file)
				fw.write("\n")
		fw.close()

##############################################################

if mode=="html_parser":
	data_collector = DataCollector()
	data_collector.getData()
elif mode=="expand":
	expander_main()
else:
	print("Wrong option")
