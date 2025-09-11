# Ejecutado desde src/
GRAMMAR = SimpleClassModelWithConstraints
ANTLR_JAR = /mnt/c/Users/dedie/Documents/LP/lpEntrega3/antlr-4.13.2-complete.jar
INPUT_FILE = ../inputs/input02.txt
MAIN_CLASS = Analizador
BUILD_DIR = comp

JAVA_FILES := $(wildcard *.java)

.PHONY: all clean generate compile run

all: generate compile

clean:
	rm -rf $(BUILD_DIR) *.tokens *.interp

generate:
	java -Xmx500M -cp "$(ANTLR_JAR)" org.antlr.v4.Tool -visitor -listener $(GRAMMAR).g4

compile:
	mkdir -p $(BUILD_DIR)
	javac -cp "$(ANTLR_JAR):." -d $(BUILD_DIR) $(JAVA_FILES)

run: all
	java -cp "$(ANTLR_JAR):$(BUILD_DIR)" $(MAIN_CLASS) $(INPUT_FILE)
