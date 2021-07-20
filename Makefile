.PHONY: clean install

PYTHON_FILES=$(wildcard CRADLE/*/*.py)
CYTHON_FILES=$(wildcard CRADLE/*/*.pyx)

clean:
	-rm -rf CRADLE.egg-info build dist
	-rm CRADLE/**/*.c

dist: $(PYTHON_FILES) $(CYTHON_FILES)
	if [ -z $$(pip list | grep -e "^build\s") ]; then pip install build; fi
	python -m build

install: dist
	pip install dist/*.whl
