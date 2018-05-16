import time

goAnnotationsPerProteinHistory = dict()
goAnnotationsPerProteinNow = dict()

lostGoAnnotations = dict()
gainedGoAnnotations = dict()

counter = 0

GOAHumanHistory = open("../data/goa_human_164.gaf", "r")
GOAHumanNow = open("../data/goa_human.gaf", "r")

lostGoAnnotationsFile = open("../data/lostGoAnnotations.txt", "w")
gainedGoAnnotationsFile = open("../data/gainedGoAnnotations.txt", "w")

print("(" + time.strftime("%c") + ")  Reading goa_human_164.gaf file...")

for line in GOAHumanHistory:
    if counter < 12:
        counter += 1
        continue
    parts = line.split("\t")
    uniprot_id = parts[1]
    goId = parts[4]

    if uniprot_id not in goAnnotationsPerProteinHistory:
        goAnnotationsPerProteinHistory[uniprot_id] = set()

    goAnnotationsPerProteinHistory[uniprot_id].add(goId)

counter = 0

print("(" + time.strftime("%c") + ")  Reading goa_human.gaf file...")

for line in GOAHumanNow:
    if counter < 1:
        counter += 1
        continue
    parts = line.split("\t")
    try:
        uniprot_id = parts[1]
        goId = parts[4]

        if uniprot_id not in goAnnotationsPerProteinNow:
            goAnnotationsPerProteinNow[uniprot_id] = set()

        goAnnotationsPerProteinNow[uniprot_id].add(goId)
    except Exception:
        print("Skipping line because of parse error: " + line)

print("(" + time.strftime("%c") + ")  Filling lost/gained dictioneries...")

for key in goAnnotationsPerProteinHistory.keys():
    historyGoAnnotations = goAnnotationsPerProteinHistory[key]
    nowGoAnnotations = set()

    if key in goAnnotationsPerProteinNow:
        nowGoAnnotations = goAnnotationsPerProteinNow[key]

    lost = historyGoAnnotations - nowGoAnnotations
    lostGoAnnotations[key] = lost

for key in goAnnotationsPerProteinNow.keys():
    nowGoAnnotations = goAnnotationsPerProteinNow[key]
    historyGoAnnotations = set()

    if key in goAnnotationsPerProteinHistory:
        historyGoAnnotations = goAnnotationsPerProteinHistory[key]

    gained = nowGoAnnotations - historyGoAnnotations
    gainedGoAnnotations[key] = gained

print("(" + time.strftime("%c") + ")  Writing into lostGoAnnotations.txt file...")

for key in lostGoAnnotations.keys():
    lostGoAnnotationsFile.write(key + " -> ")
    if len(lostGoAnnotations[key]) == 0:
        lostGoAnnotationsFile.write("{}")
    else:
        lostGoAnnotationsFile.write(str(lostGoAnnotations[key]))
    lostGoAnnotationsFile.write("\t\n")

print("(" + time.strftime("%c") + ")  Writing into gainedGoAnnotations.txt file...")

for key in gainedGoAnnotations.keys():
    gainedGoAnnotationsFile.write(key + " -> ")
    if len(gainedGoAnnotations[key]) == 0:
        gainedGoAnnotationsFile.write("{}")
    else:
        gainedGoAnnotationsFile.write(str(gainedGoAnnotations[key]))
    gainedGoAnnotationsFile.write("\t\n")

lostGoAnnotationsFile.close()
gainedGoAnnotationsFile.close()
GOAHumanHistory.close()
GOAHumanNow.close()
