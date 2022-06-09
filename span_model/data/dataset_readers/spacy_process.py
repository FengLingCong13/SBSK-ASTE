import spacy
import networkx as nx


def spacy_preprocess(sentence_str, span_len):
    span = []
    for i in range(len(sentence_str)):
        if i + span_len <= len(sentence_str):
            for j in range(span_len):
                span.append(sentence_str[i:i + j + 1])
        else:
            for j in range(len(sentence_str) - i):
                span.append(sentence_str[i:i + j + 1])
    span_results = []
    for i in range(len(span)):
        span_temp = ''
        for j in range(len(span[i])):
            span_temp = span_temp + span[i][j] + ' '
        span_temp = span_temp.strip()
        span_results.append(span_temp)

    sentence = ''
    for i in range(len(sentence_str)):
        sentence = sentence + str(sentence_str[i]) + ' '
    sentence = sentence.strip()

    return sentence, span_results


def spacy_process(sentence, aspect_and_opinion, threshold):
    matrix = [[1] * len(aspect_and_opinion) for _ in range(len(aspect_and_opinion))]
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
    except NameError as e:
        raise RuntimeError('Fail to load nlp model, maybe you forget to download en_core_web_sm')

    edges = []
    sentence2 = ''
    for token in doc:
        sentence2 = sentence2 + str(token) + ' '
    sentence2 = sentence2.strip()

    aspect_terms = []
    for m in range(len(aspect_and_opinion)):
        aspect_term_temp = ''
        doc = nlp(aspect_and_opinion[m])
        for token in doc:
            aspect_term_temp = aspect_term_temp + str(token) + ' '
        aspect_term_temp = aspect_term_temp.strip()
        aspect_terms.append(aspect_term_temp)

    opinion_terms = aspect_terms

    doc = nlp(sentence2)

    for token in doc:
        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_, token.i),
                          '{}_{}'.format(child.lower_, child.i)))
    graph = nx.Graph(edges)

    for m in range(len(aspect_terms)):
        for n in range(m, len(opinion_terms)):
            if m == n:
                matrix[m][m] = 0
                continue
            aspect_term = aspect_terms[m]
            opinion_term = opinion_terms[n]
            aspects = [a.lower() for a in aspect_term.split()]
            opinions = [a.lower() for a in opinion_term.split()]

            # Load spacy's dependency tree into a networkx graph

            cnt_aspect = 0
            cnt_opinion = 0
            aspect_ids = [0] * len(aspects)
            opinion_ids = [0] * len(opinions)
            for token in doc:
                # Record the position of aspect terms
                if cnt_aspect < len(aspects) and token.lower_ == aspects[cnt_aspect]:
                    aspect_ids[cnt_aspect] = token.i

                    cnt_aspect += 1
                # Record the position of opinion terms
                if cnt_opinion < len(opinions) and token.lower_ == opinions[cnt_opinion]:
                    opinion_ids[cnt_opinion] = token.i
                    cnt_opinion += 1

            if len(aspect_terms) == 1 and len(opinion_terms) == 1:
                return matrix

            dist = [0.0] * len(doc)
            for i, word in enumerate(doc):
                source = '{}_{}'.format(word.lower_, word.i)
                sum = 0
                max_dist = 0
                for aspect_id, aspect in zip(aspect_ids, aspects):
                    target = '{}_{}'.format(aspect, aspect_id)
                    try:
                        sum += nx.shortest_path_length(graph, source=source, target=target)
                    except:
                        sum += len(doc)  # No connection between source and target
                        flag = 0
                dist[i] = sum / len(aspects)

            aspect_opinion_dist = 0

            for i in range(len(opinions)):
                aspect_opinion_dist += dist[opinion_ids[i]]
            if len(opinions) == 0:
                continue

            aspect_opinion_dist = aspect_opinion_dist / len(opinions)

            if aspect_opinion_dist > threshold:
                matrix[m][n] = 0
                matrix[n][m] = 0
    return matrix

# sentence, span_results = spacy_preprocess(sentence_str, 8)
#
# matrix = spacy_process(sentence, span_results, 5)
