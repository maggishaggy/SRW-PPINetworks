from __future__ import print_function

import time
import fastsemsim
import numpy as np
from fastsemsim import SemSim


def load_ontology():
    return fastsemsim.Ontology.ontologies.load(source='Data/go_daily-termdb.obo-xml', source_type='obo-xml')


def load_custom_ontology(type_='BP'):
    terms = dict()
    edges = list()
    t = set()
    if type_ == 'BP':
        file_name = 'Data/BPGOfull.txt'
    elif type_ == 'CC':
        file_name = 'Data/CCGOfull.txt'
    else:
        file_name = 'Data/MFGOfull.txt'
    with open(file_name, 'r') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n')
            parts = line.split(' ')
            t.add(parts[0])
            t.add(parts[2])
            edges.append([parts[0], parts[2], parts[1]])
            line = doc.readline()
    terms['id'] = list(t)
    terms['alt_id'] = []
    terms['replaced_by'] = []
    terms['is_obsolete'] = []
    terms['namespace'] = []
    terms['name'] = []
    terms['def'] = []
    return fastsemsim.Ontology.GeneOntology.GeneOntology(terms=terms, edges=edges, parameters=None)


def load_annotation_corpus(go, file_name):
    annotations = dict()
    reverse_annotations = dict()
    obj_set = set()
    term_set = set()
    with open(file_name, 'r') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n')
            line = line.replace('GO:', '')
            parts = line.split('\t')
            go_id = int(parts[2])
            if parts[0] in annotations.keys():
                tmp = annotations[parts[0]]
                tmp.append(go_id)
                annotations[parts[0]] = tmp
            else:
                annotations[parts[0]] = [go_id]
            if go_id in reverse_annotations.keys():
                tmp = reverse_annotations[go_id]
                tmp.append(parts[0])
                reverse_annotations[go_id] = tmp
            else:
                reverse_annotations[go_id] = [parts[0]]
            obj_set.add(parts[0])
            term_set.add(go_id)
            line = doc.readline()
    annotation_corpus = fastsemsim.Ontology.AnnotationCorpus.AnnotationCorpus(go)
    annotation_corpus.annotations = annotations
    annotation_corpus.reverse_annotations = reverse_annotations
    annotation_corpus.obj_set = obj_set
    annotation_corpus.term_set = term_set
    return annotation_corpus


def compute_matrix(terms, measure, go, is_resnik=False):
    matrix = np.zeros((len(terms), len(terms)))
    start = time.clock()
    max_value = max(measure.util.IC.values())
    for i in range(len(terms)):
        for j in range(i, len(terms)):
            val = measure.SemSim(terms[i], terms[j], go)
            if is_resnik:
                val = val / max_value
            val = round(val, 5)
            matrix[i][j] = val
            matrix[j][i] = val
    elapsed = time.clock() - start
    print('Number of terms: ' + str(len(terms)))
    print('Elapsed time: ' + str(round(elapsed, 5)))
    return matrix


def save(file_name, terms, matrix):
    with open(file_name, 'w+') as doc:
        doc.write('\t')
        for term in terms:
            doc.write('GO:' + format(term, '07d') + '\t')
        doc.write('\n')
        for i in range(len(terms)):
            doc.write('GO:' + format(terms[i], '07d') + '\t')
            for j in range(len(terms)):
                doc.write(str(matrix[i][j]) + '\t')
            doc.write('\n')
    return


ppi_700_go_bp = 'Data/HumanPPI700_GO_BP_modified.txt'
ppi_700_go_cc = 'Data/HumanPPI700_GO_CC_modified.txt'
ppi_700_go_mf = 'Data/HumanPPI700_GO_MF_modified.txt'
ppi_700_go_bp_terms_resnik = 'Data/HumanPPI700_GO_BP_modified_terms_resnik.txt'
ppi_700_go_cc_terms_resnik = 'Data/HumanPPI700_GO_CC_modified_terms_resnik.txt'
ppi_700_go_mf_terms_resnik = 'Data/HumanPPI700_GO_MF_modified_terms_resnik.txt'
ppi_700_go_bp_terms_lin = 'Data/HumanPPI700_GO_BP_modified_terms_lin.txt'
ppi_700_go_cc_terms_lin = 'Data/HumanPPI700_GO_CC_modified_terms_lin.txt'
ppi_700_go_mf_terms_lin = 'Data/HumanPPI700_GO_MF_modified_terms_lin.txt'
ppi_900_go_bp = 'Data/HumanPPI900_GO_BP_modified.txt'
ppi_900_go_cc = 'Data/HumanPPI900_GO_CC_modified.txt'
ppi_900_go_mf = 'Data/HumanPPI900_GO_MF_modified.txt'
ppi_900_go_bp_terms_resnik = 'Data/HumanPPI900_GO_BP_modified_terms_resnik.txt'
ppi_900_go_cc_terms_resnik = 'Data/HumanPPI900_GO_CC_modified_terms_resnik.txt'
ppi_900_go_mf_terms_resnik = 'Data/HumanPPI900_GO_MF_modified_terms_resnik.txt'
ppi_900_go_bp_terms_lin = 'Data/HumanPPI900_GO_BP_modified_terms_lin.txt'
ppi_900_go_cc_terms_lin = 'Data/HumanPPI900_GO_CC_modified_terms_lin.txt'
ppi_900_go_mf_terms_lin = 'Data/HumanPPI900_GO_MF_modified_terms_lin.txt'
# ontology = load_ontology()

print('===========700')

ontology = load_custom_ontology('BP')
ac = load_annotation_corpus(ontology, ppi_700_go_bp)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('============BP')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_700_go_bp_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_700_go_bp_terms_lin, go_terms, similarity)

ontology = load_custom_ontology('CC')
ac = load_annotation_corpus(ontology, ppi_700_go_cc)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('\n============CC')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_700_go_cc_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_700_go_cc_terms_lin, go_terms, similarity)

ontology = load_custom_ontology('MF')
ac = load_annotation_corpus(ontology, ppi_700_go_mf)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('\n============MF')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_700_go_mf_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_700_go_mf_terms_lin, go_terms, similarity)


print('\n\n===========900')

ontology = load_custom_ontology('BP')
ac = load_annotation_corpus(ontology, ppi_900_go_bp)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('============BP')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_900_go_bp_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_900_go_bp_terms_lin, go_terms, similarity)

ontology = load_custom_ontology('CC')
ac = load_annotation_corpus(ontology, ppi_900_go_cc)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('\n============CC')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_900_go_cc_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_900_go_cc_terms_lin, go_terms, similarity)

ontology = load_custom_ontology('MF')
ac = load_annotation_corpus(ontology, ppi_900_go_mf)
go_terms = ac.reverse_annotations.keys()
go_terms = sorted(go_terms)
resnik = SemSim.ResnikSemSim(ontology, ac)
lin = SemSim.LinSemSim(ontology, ac)
print('\n============MF')
print(' =======Resnik')
similarity = compute_matrix(go_terms, resnik, ontology, True)
save(ppi_900_go_mf_terms_resnik, go_terms, similarity)
print(' ==========Lin')
similarity = compute_matrix(go_terms, lin, ontology)
save(ppi_900_go_mf_terms_lin, go_terms, similarity)
