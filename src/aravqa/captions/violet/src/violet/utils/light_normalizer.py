'''
Author: AbdelRahim Elmadany, Muhammad Abdul-Mageed
Copyright: All rights are reserved to University of British Columbia (UBC), NLP LAP, 2020
Last update: March, 2020
'''
# -*- coding: utf-8 -*-
import os, sys
if sys.version_info[0] < 3:
    raise Exception("Error: This code based on Python 3")
    
#from pyspark.sql.types import StringType, StructField, StructType, BooleanType, ArrayType, IntegerType, LongType, Row

import regex, re

class light_normalizer():
	"""A class for shared functions"""
    # Initializing 
	def __init__(self):
		#print ("Loading Light Normalizer (depending on lighting conditions)... ", end=' ') 
		self.Arabic_normalized_chars= {u"\u0622":u"\u0627", u"\u0623":u"\u0627", u"\u0625":u"\u0627", u"\u0649":u"\u064A", u"\u0629":u"\u0647"}
		#print ("Done")
	def normalizeArabicChar(self,inputText):
		norm=""
		for char in inputText:
			if char in self.Arabic_normalized_chars:
				norm = norm + self.Arabic_normalized_chars[char]
			else:
				norm = norm + char
		return norm
	
	def run_light_normalizer(self,tweet, cased=False):
		norm_text=""
		words_text=""
		num_words=0
		if tweet and str(tweet).strip():
			norm_text = re.sub('"', '\'', str(tweet)).strip()#.decode('utf8')
			# norm_text = re.sub('[\n\r\f]+', ' ', str(norm_text))+" "  # new lines
			#--for Arabic
			if cased:
				norm_text = self.normalizeArabicChar(norm_text)
			
			#remove Tashkeel
			#normalized_tweet = self.remover_tashkeel(normalized_tweet)
			norm_text = regex.sub(r'[ـًٌٍَُِّْْ]+',"", norm_text)
			#replace URL, USR
			norm_text = re.sub("(http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#-[\]@!\$&'\(\)\*\+,;=.]+"," URL ", norm_text, re.UNICODE) #URL
			# norm_text = regex.sub(r'@[\p{N}\p{L}_]+'," USER ", norm_text)
			#normalized_tweet = regex.sub(r'(\p{N}+| \p{N}+\.\p{N}+)'," NUM ", normalized_tweet)
			#normalized_tweet = regex.sub(r'[\p{P}\p{C}\p{S}]+'," ", normalized_tweet) #punctuation,  invisible control characters, symbols
			# norm_text = regex.sub(r'\p{C}+'," ", norm_text) #invisible control characters

			###################################################################################################################

			# pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
			# norm_text = pattern.sub(r"\1\1", norm_text) #Reduce character repitation of > 2 characters at time 
			# pattern = re.compile(r"(..)\1{2,}", re.DOTALL)
			# norm_text = pattern.sub(r"\1", norm_text) #Reduce character repitation of > 2 characters at time 
			
			
			# # norm_text = regex.sub(r'\p{Z}+'," ", norm_text) #any kind of whitespace or invisible separator.
			
			# norm_text = regex.sub(r'(\p{L}\s)\1{1,}',r"\1", norm_text) #Reduce word repitation of > 1 at time 
			
			# norm_text = re.sub("&(?:amp|lt|gt|quot);"," ", norm_text) #remove html code &..
			# #cnvert hash to text such as #one_two --> one two
			# norm_text = regex.sub(r'#',"", norm_text)
			# norm_text = regex.sub(r'_'," ", norm_text)

			#################################################################################################################
			#------------------------------
			# words_text = regex.sub(r'[^\p{L}\s]',"", norm_text)
			# words_text = regex.sub(r'URL',"", words_text)
			# num_words = len(words_text.strip().split(" "))
		return str(norm_text).strip()#, str(words_text).strip(), num_words]

# ~ def run(f, fout, cased):
	# ~ norm = Normalizer()
	# ~ c=0
	# ~ doc=""
	# ~ with open(f,"r") as inp:
		# ~ for line in inp:
			# ~ nom_txt= norm.run_lite_normalizer(line.strip(), cased)
			# ~ doc+= nom_txt+"\n"
			# ~ c+=1
			# ~ #print (empty)
			# ~ #print ("\r#lines are done are "+str(c), end="")
	
	# ~ print ("\n Reduce newline repitation of > 2 at time")
	# ~ doc = regex.sub(r'([\r\n\f]+)\1{2,}',r"\1", doc) #Reduce newline repitation of > 2 at time 
	# ~ print ("\n saving output file")
	# ~ with open(fout,"w") as fout_obj:
		# ~ fout_obj.write(doc)
		# ~ fout_obj.flush()
	# ~ print ("\nDone All")
# ~ if __name__ == "__main__":
	# ~ cased=True	
	# ~ fout=""
	# ~ cased_arg = sys.argv[2]
	# ~ print (cased_arg)
	# ~ if "cased" == cased_arg:
		# ~ fout= sys.argv[1]+".norm_cased"
		# ~ cased=True
	# ~ elif "uncased" == cased_arg:
		# ~ fout= sys.argv[1]+".norm_uncased"
		# ~ cased=False
	# ~ print (cased, fout)
	# ~ run(sys.argv[1], fout, cased)
