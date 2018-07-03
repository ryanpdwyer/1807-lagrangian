


freeze: requirements-grep
	pip freeze | grep -f requirements-grep.txt > requirements-frozen.txt
	rm requirements-grep.txt

requirements-grep:
	sed 's/#.*$$//' requirements.txt > requirements-grep.txt
