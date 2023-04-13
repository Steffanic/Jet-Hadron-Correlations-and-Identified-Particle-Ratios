doAnalysis: doAnalysis.py
	python3 doAnalysis.py

swapon: 
	sudo swapon /media/swapfile.img
	sudo swapon /media/swapfile_ext.img

clean:
	rm -r ../backend_output/*/*.png
	rm -r ../backend_output/*/*/*.png