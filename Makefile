doAnalysis: doAnalysis.py
	python3 doAnalysis.py

deletePickles: 
	-rm jhAnapp.pickle 
	-rm jhAnaCentral.pickle
	-rm jhAnaSemiCentral.pickle

agePickles:
	-mv jhAnapp.pickle jhAnappOLD.pickle
	-mv jhAnaCentral.pickle jhAnaCentralOLD.pickle
	-mv jhAnaSemiCentral.pickle jhAnaSemiCentralOLD.pickle

swapon: 
	sudo swapon /media/swapfile.img
	sudo swapon /media/swapfile_ext.img

clean:
	rm -r ../backend_output/*/*.png
	rm -r ../backend_output/*/*/*.png
