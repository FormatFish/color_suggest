#coding=utf-8
import matlab.engine

def ratingPalette(input_file , out= 'tmp.txt'):
	eng = matlab.engine.start_matlab()
	res = eng.evaluateFun(input_file , out)
	rating = [item[0] for item in res]
	return rating
