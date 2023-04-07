doAnalysis: doAnalysis.py
	python3 doAnalysis.py

clean:
	rm -r ../backend_output/*/*.png
	rm -r ../backend_output/*/*/*.png